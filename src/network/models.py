from collections import OrderedDict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet101, vgg11
from torchsummary import summary
import torchvision
import numpy as np
import cv2

from network.PIXIE.pixielib.pixie import PIXIE
from network.PIXIE.pixielib.utils.config import cfg as pixie_cfg
from network.PIXIE.pixielib.datasets import detectors
from utils.image_process import crop_resize_image_batch
from network.sub_module.modules import CNN_Encoder, LSTM_Decoder, GRU_Decoder, RNN_Decoder
from network.sub_module.smplx_joint_ids2names import selected_indicecs
from utils.vis_3d import plot_and_save_point_cloud, plotly_save_point_cloud

class CNN_LSTM(nn.Module):
    """CNN-LSTM model combining CNN and LSTM components."""
    def __init__(self, 
                 n_classes, 
                 latent_size, 
                 n_cnn_layers, 
                 n_rnn_layers, 
                 n_rnn_hidden_dim,
                 channel_in=3, 
                 cnn_act_fn="relu", 
                 rnn_act_fn="relu", 
                 dropout_rate=0.1,
                 cnn_bn=False, 
                 bidirectional=False, 
                 attention=False,
                 device="cuda",
                 use_2d_kps=False,
                 use_3d_kps = False,
                 use_img_feats = False,
                 body_detector = 'rcnn',
                 input_size = 256):
        
        super(CNN_LSTM, self).__init__()

        self.attention = attention
        
        self.use_2d_kps = use_2d_kps
        self.use_3d_kps = use_3d_kps
        self.use_img_feats = use_img_feats

        self.num_2d_keypoints = 17
        self.num_3d_keypoints = len(selected_indicecs)
        self.input_size = input_size
        self.pixie_input_size = 224
        self.scale = 1.1

        if use_img_feats:
            lstm_latent_size = latent_size
        else:
            lstm_latent_size = 0
            
        if self.use_2d_kps:
            lstm_latent_size +=  3 * self.num_2d_keypoints
        if self.use_3d_kps:
            lstm_latent_size +=  3 * self.num_3d_keypoints

        if use_img_feats:
            self.encoder = CNN_Encoder(latent_size, 
                                    n_cnn_layers, 
                                    intermediate_act_fn=cnn_act_fn, 
                                    use_batchnorm=cnn_bn, 
                                    channel_in=channel_in)
        else:
            self.encoder = None
        self.decoder = LSTM_Decoder(lstm_latent_size, 
                                    n_rnn_hidden_dim, 
                                    n_classes, 
                                    n_rnn_layers, 
                                    intermediate_act_fn=rnn_act_fn, 
                                    bidirectional=bidirectional, 
                                    attention=attention,
                                    device=device)
        self.device = device

        if use_2d_kps:
            self.rcnn_keypoints_model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
            self.rcnn_keypoints_model.eval()
            self.rcnn_keypoints_model.to(device)

            ## freeze keypoint RCNN all parameters
            for param in self.rcnn_keypoints_model.parameters():
                param.requires_grad = False

        if use_3d_kps:
            print('use 3d keypoints: {}'.format(use_3d_kps))
            self.selected_indicecs = torch.tensor(selected_indicecs, device=self.device)
            self.pixie_model = PIXIE(config = pixie_cfg, device=device, freeze_model=True)
            if body_detector == 'rcnn':
                self.detector = detectors.FasterRCNN(device=device)
            elif body_detector == 'keypoint':
                self.detector = detectors.KeypointRCNN(device=device)
            else:
                print('no detector is used')

            # self.Encoder, self.Regressor, self.Moderator, self.Extractor
            for param in self.detector.model.parameters():
                param.requires_grad = False
            

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # print(x.size())  # torch.Size([batch_size, n_frames, 3, 256, 256])
        # print(f'x min: {x.min()}, max: {x.max()}') #min: 0.0, max: 1.0
        batch_size, timesteps, C, H, W = x.size() # torch.Size([batch_size, 30, 3, 256, 256])

        x_reshaped = x.view(batch_size * timesteps, C, H, W)
        # print(f'x_reshaped min: {x_reshaped.min()}, max: {x_reshaped.max()}') #min: 0.0, max: 1.0
        # resize image to input size
        if H != self.input_size or W != self.input_size:
            x_reshaped_resized = F.interpolate(x_reshaped, size=(self.input_size, self.input_size), mode='bilinear')
        # print(f'x_reshaped_resized min: {x_reshaped_resized.min()}, max: {x_reshaped_resized.max()}') #min: 0.0, max: 1.0
        # print(x_reshaped_resized.size())  # torch.Size([batch_size*n_frames, 3, 256, 256])

        if self.use_img_feats:
            # print(latent_var.size())  # torch.Size([120, 512]
            latent_var = self.encoder(x_reshaped_resized) # torch.Size([120, 512]

            rnn_in = latent_var.view(batch_size, timesteps, -1)

        if self.use_2d_kps:
            #print('x_reshaped.size()', x_reshaped.size())
            # Use KeypointRCNN for keypoint detection
            self.rcnn_keypoints_model.eval()
            if hasattr(self, 'rcnn_keypoints_model'):
                with torch.no_grad():  # Ensure no gradients are calculated
                    results = self.rcnn_keypoints_model(x_reshaped_resized)
                    keypoints_tensors = []
                    for result in results:
                        if len(result['scores']) > 0:
                            max_score_index = result['scores'].argmax()
                            keypoints = result['keypoints'][max_score_index]  # [num_keypoints, 3]
                            if keypoints.shape[0] < self.num_2d_keypoints:
                                pad_size = self.num_2d_keypoints - keypoints.shape[0]
                                pad = torch.zeros((pad_size, 3), device=keypoints.device)
                                keypoints = torch.cat([keypoints, pad], dim=0)
                            keypoints_tensors.append(keypoints)
                        else:
                            keypoints_tensors.append(torch.zeros((self.num_2d_keypoints, 3), device=self.device))

                    keypoints_tensor = torch.stack(keypoints_tensors).view(batch_size, timesteps, -1)
                    if self.use_img_feats:
                        rnn_in = torch.cat((rnn_in, keypoints_tensor), dim=2)
                    else:
                        rnn_in = keypoints_tensor

        if self.use_3d_kps:
            batch = {}
            with torch.no_grad():
                batch['image_hd'] = x_reshaped
                
                _, c, h_1, w_1 = x_reshaped_resized.size()
                _, c, hd_h, hd_w = x_reshaped.size()
                scale_x = hd_w / w_1
                slice_y = hd_h / h_1
                bbox = self.detector.run(x_reshaped_resized[0:1])
                temp_img = x_reshaped_resized[0:1].cpu().numpy().transpose(0, 2, 3, 1).squeeze()
                # print('temp_img, min, max', temp_img.min(), temp_img.max())
                # cv2.imwrite('temp_img.jpg', (temp_img*255).astype(np.uint8))
                # print('bbox', bbox) # [104.55945  46.06325 168.01297 255.5114 ]
                bboxes = self.detector.run_batch(x_reshaped_resized)
                if isinstance(bboxes, np.ndarray):
                    bboxes = torch.tensor(bboxes, device=self.device)
                # print('1 bboxes', bboxes)
                
                # print('bboxes.size()', bboxes.size())  # torch.Size([bs*n_frames, 4])
                bboxes_new = bboxes * torch.tensor([scale_x, slice_y, scale_x, slice_y], device=self.device)
                
                # print('bboxes_new.size()', bboxes_new.size())  # torch.Size([bs*n_frames, 4])
                # print('2 bboxes', bboxes)
                # print('bboxes_new', bboxes_new)

                crop_resized_image = crop_resize_image_batch(x_reshaped, bboxes_new, self.scale, self.pixie_input_size)
                batch['image'] = crop_resized_image
                # batch['name'] = 'sign_image'
                # name = batch['name']
                # name = os.path.basename(name)
                # name = name.split('.')[0]
                # print(name)
                # frame_id = int(name.split('frame')[-1])
                # name = f'{frame_id:05}'

                data = {
                    'body': batch
                }
            
                param_dict = self.pixie_model.encode(data, threthold=True, keep_local=True, copy_and_paste=False)
                
                # param_dict = pixie.encode(data, threthold=True, keep_local=True, copy_and_paste=True)
                # only use body params to get smplx output. TODO: we can also show the results of cropped head/hands
                codedict = param_dict['body']

                prediction = self.pixie_model.decode(codedict, param_type='body', skip_pytorch3d = True)

                joints_3d = prediction['joints']   # shape: [bs, 145, 3]
                
                joints_3d = joints_3d[:, self.selected_indicecs, :]
                # print('joints_3d.size()', joints_3d.size())  # torch.Size([bs, 17, 3])
                # plotly_save_point_cloud(joints_3d[0].cpu().numpy())
                joints_3d = joints_3d.reshape(batch_size, timesteps, -1)

                if self.use_img_feats:
                    rnn_in = torch.cat((rnn_in, joints_3d), dim=2)
                else:
                    rnn_in = joints_3d

        
        rnn_in = self.dropout(rnn_in)
        out, _ = self.decoder(rnn_in)

        return out



class VGG_LSTM(nn.Module):
    """VGG-LSTM model using VGG11 as CNN component."""
    def __init__(self, 
                 n_classes, 
                 latent_size, 
                 n_rnn_layers,
                 n_rnn_hidden_dim,
                 rnn_act_fn="relu",
                 dropout_rate=0.1,
                 bidirectional=False,
                 device="cuda",
                 use_2d_kps=False,
                 use_3d_kps = False,
                 use_img_feats = False,
                 body_detector = 'rcnn',
                 input_size = 256):
        
        super(VGG_LSTM, self).__init__()
        
        self.CNN = vgg11(pretrained=True, progress=False)
        self.CNN.classifier[6] = nn.Linear(4096, latent_size)

        self.decoder = LSTM_Decoder(latent_size, n_rnn_hidden_dim, n_classes, n_rnn_layers, intermediate_act_fn=rnn_act_fn, bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.input_size = input_size    

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size() # torch.Size([batch_size, 30, 3, 256, 256])

        x_reshaped = x.view(batch_size * timesteps, C, H, W)
        
        # resize image to input size
        if H != self.input_size or W != self.input_size:
            x_reshaped_resized = F.interpolate(x_reshaped, size=(self.input_size, self.input_size), mode='bilinear')

        latent_var = self.CNN(x_reshaped_resized)

        rnn_in = latent_var.view(batch_size, timesteps, -1)
        rnn_in = self.dropout(rnn_in)
        out, _ = self.decoder(rnn_in)
        # print('out.size()', out.size()) # torch.Size([16, 10])
        # print('out[0].size()', out[0].size()) # torch.Size([10])

        return out

# extra models for experimentation below



class CNN_GRU(nn.Module):

    def __init__(self, 
                 n_classes, 
                 latent_size, 
                 n_cnn_layers, 
                 n_rnn_layers, 
                 n_rnn_hidden_dim,
                 channel_in=3, 
                 cnn_act_fn="relu", 
                 rnn_act_fn="relu", 
                 dropout_rate=0.1,
                 cnn_bn=False, 
                 bidirectional=False):
        
        super(CNN_GRU, self).__init__()

        self.encoder = CNN_Encoder(latent_size, n_cnn_layers, intermediate_act_fn=cnn_act_fn, use_batchnorm=cnn_bn, channel_in=channel_in)
        self.decoder = GRU_Decoder(latent_size, n_rnn_hidden_dim, n_classes, n_rnn_layers, intermediate_act_fn=rnn_act_fn, bidirectional=bidirectional)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):

        batch_size, timesteps, C, H, W = x.size()

        cnn_in = x.view(batch_size * timesteps, C, H, W)
        latent_var = self.encoder(cnn_in)

        rnn_in = latent_var.view(batch_size, timesteps, -1)
        rnn_in = self.dropout(rnn_in)
        out = self.decoder(rnn_in)

        return out

class CNN_RNN(nn.Module):

    def __init__(self, 
                 n_classes, 
                 latent_size, 
                 n_cnn_layers, 
                 n_rnn_layers, 
                 n_rnn_hidden_dim,
                 channel_in=3, 
                 cnn_act_fn="relu", 
                 rnn_act_fn="relu", 
                 dropout_rate=0.1,
                 cnn_bn=False, 
                 bidirectional=False):
        
        super(CNN_RNN, self).__init__()

        self.encoder = CNN_Encoder(latent_size, n_cnn_layers, intermediate_act_fn=cnn_act_fn, use_batchnorm=cnn_bn, channel_in=channel_in)
        self.decoder = RNN_Decoder(latent_size, n_rnn_hidden_dim, n_classes, n_rnn_layers, intermediate_act_fn=rnn_act_fn, bidirectional=bidirectional)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):

        batch_size, timesteps, C, H, W = x.size()

        cnn_in = x.view(batch_size * timesteps, C, H, W)
        latent_var = self.encoder(cnn_in)

        rnn_in = latent_var.view(batch_size, timesteps, -1)
        rnn_in = self.dropout(rnn_in)
        out = self.decoder(rnn_in)

        return out

class ResNet_LSTM(nn.Module):

    def __init__(self, 
                 n_classes, 
                 latent_size, 
                 n_rnn_layers, 
                 channel_in=3, 
                 rnn_act_fn="relu", 
                 bidirectional=False, 
                 resnet_opt="resnet18"):
        
        super(ResNet_LSTM, self).__init__()

        if resnet_opt == "resnet18":
            self.CNN = resnet18(pretrained=True)
        elif resnet_opt == "resnet101":
            self.CNN = resnet101(pretrained=True)
        
        self.CNN.fc = nn.Sequential(nn.Linear(self.CNN.fc.in_features, latent_size))

        self.LSTM = LSTM_Decoder(latent_size, 3, n_classes, n_rnn_layers, intermediate_act_fn=rnn_act_fn, bidirectional=bidirectional)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()

        cnn_in = x.view(batch_size * timesteps, C, H, W)
        latent_var = self.CNN(cnn_in)

        rnn_in = latent_var.view(batch_size, timesteps, -1)
        out = self.RNN(rnn_in)

        out = self.softmax(out)
        return out


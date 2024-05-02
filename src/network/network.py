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
import matplotlib.pyplot as plt

from network.sub_module.encoder_CNN import CNN_Encoder
from network.sub_module.encoder_2D_kpts import Encoder_2D_kpts
from network.sub_module.encoder_3D_kpts import Encoder_3D_kpts

from network.sub_module.decoder_transformers import TransformerDecoder, TransformerFull, TransformerEncoderCls
from network.sub_module.decoder_RNNs import LSTM_Decoder, GRU_Decoder, RNN_Decoder

class SLR_network(nn.Module):
    def __init__(self, 
                 n_classes,                
                 device="cpu",
                 encoder_types = None,
                 decoder_types = None,
                 body_detector = 'rcnn',
                 trans_num_layers = 4,
                 input_size = 256):
        
        super(SLR_network, self).__init__()

        channel_in=3
        cnn_act_fn="relu"
        rnn_act_fn="relu"

        n_cnn_layers=6
        n_rnn_layers=1
        n_rnn_hidden_dim=512
        cnn_bn=True
        bidirectional=True
        dropout_rate=0.8
        attention=True

        self.device = device
        self.attention = attention
        
        self.encoder_types = encoder_types
        self.decoder_types = decoder_types

        self.input_size = input_size
        encoder_ouput_channel = 0

        if encoder_types is None:
            raise ValueError("Please provide encoder types")
        encoder_list = ['CNN', '2D_kpts', '3D_kpts'] 
        for encoder_name in encoder_types:
            assert encoder_name in encoder_list , f"encoder_name should be one of {encoder_list}"
            if encoder_name == 'CNN':
                self.encoder = CNN_Encoder(
                                        n_layers = n_cnn_layers, 
                                        intermediate_act_fn=cnn_act_fn, 
                                        use_batchnorm=cnn_bn, 
                                        channel_in=channel_in)
                encoder_ouput_channel += self.encoder.channel_out
                self.use_img_feats = True
            else:
                self.use_img_feats = False

            if encoder_name == '2D_kpts':
                self.encoder_2d_kps = Encoder_2D_kpts(freeze_weights=True, load_pretrained = True, device=device)
                encoder_ouput_channel +=  3 * self.encoder_2d_kps.num_2d_keypoints
                self.use_2d_kps = True
            else:
                self.use_2d_kps = False

            if encoder_name == '3D_kpts':
                self.encoder_3d_kps = Encoder_3D_kpts(freeze_weights=True, load_pretrained = True, device=device, body_detector = body_detector)
                encoder_ouput_channel +=  3 * self.encoder_3d_kps.num_3d_keypoints
                self.use_3d_kps = True
            else:
                self.use_3d_kps = False

        nhead = 3
        assert encoder_ouput_channel % nhead == 0, "encoder_ouput_channel should be divisible by nhead"

        if decoder_types is None:
            raise ValueError("Please provide decoder types")
        
        decoder_list = ['TransformerDecoder', 'TransformerFull', 'TransformerEncoderCls', 'LSTM']

        for decoder_name in decoder_types:
            assert decoder_name in decoder_list , f"decoder_name should be one of {decoder_list}"
            if decoder_name == 'TransformerDecoder':
                self.decoder = TransformerDecoder(
                                            n_classes,
                                            encoder_ouput_channel, 
                                            nhead,
                                            num_layers = trans_num_layers,
                                            device=device)
            

            if decoder_name == 'TransformerFull':
                self.decoder = TransformerFull(
                                            n_classes,
                                            encoder_ouput_channel, 
                                            num_layers = trans_num_layers,
                                            device=device)

            if decoder_name == 'TransformerEncoderCls':
                self.decoder = TransformerEncoderCls(
                                            n_classes,
                                            encoder_ouput_channel, 
                                            num_layers = trans_num_layers,
                                            device=device)
                
            if decoder_name == 'LSTM':
                self.decoder = LSTM_Decoder(encoder_ouput_channel, 
                                            n_rnn_hidden_dim, 
                                            n_classes, 
                                            n_rnn_layers, 
                                            intermediate_act_fn=rnn_act_fn, 
                                            bidirectional=bidirectional, 
                                            attention=attention,
                                            device=device)
                
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

            decoder_in = latent_var.view(batch_size, timesteps, -1)

        if self.use_2d_kps:
            #print('x_reshaped.size()', x_reshaped.size())
            # Use KeypointRCNN for keypoint detection
            keypoints_tensor = self.encoder_2d_kps(x_reshaped_resized)  # torch.Size([bs*n_frames, 17, 3])
            keypoints_tensor = keypoints_tensor.view(batch_size, timesteps, -1)

            if self.use_img_feats:
                decoder_in = torch.cat((decoder_in, keypoints_tensor), dim=2)
            else:
                decoder_in = keypoints_tensor

        if self.use_3d_kps:
            joints_3d = self.encoder_3d_kps(x_reshaped, x_reshaped_resized)  # torch.Size([bs, 17, 3])
            # print('joints_3d.size()', joints_3d.size())  # torch.Size([bs, 17, 3])
            # plotly_save_point_cloud(joints_3d[0].cpu().numpy())
            joints_3d = joints_3d.reshape(batch_size, timesteps, -1)

            if self.use_img_feats:
                decoder_in = torch.cat((decoder_in, joints_3d), dim=2)
            else:
                decoder_in = joints_3d

        
        decoder_in = self.dropout(decoder_in)
        classification, _ = self.decoder(decoder_in)

        return classification



if __name__ == "__main__":    
    pass
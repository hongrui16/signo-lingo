from collections import OrderedDict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

from network.PIXIE.pixie_zoo.pixie import PIXIE
from network.PIXIE.pixie_zoo.utils.config import cfg as pixie_cfg
from network.PIXIE.pixie_zoo.datasets import detectors
from network.sub_module.SMPLX_joint_ids2names import selected_indicecs

from utils.vis_3d import plot_and_save_point_cloud, plotly_save_point_cloud
from utils.image_process import crop_resize_image_batch

class Compute_3D_kpts(nn.Module):
    """3D Keypoints Encoder for 3D keypoints part of model."""
    def __init__(self, freeze_weights=True, load_pretrained = True, device='cuda', body_detector = 'rcnn', no_detector = False):

        super(Compute_3D_kpts, self).__init__()

        self.device = device
        self.selected_indicecs = selected_indicecs
        self.body_detector = body_detector
        self.num_3d_keypoints = len(selected_indicecs)
        self.pixie_input_size = 224
        self.scale = 1.1


        self.selected_indicecs = torch.tensor(selected_indicecs, device=self.device)
        self.pixie_model = PIXIE(config = pixie_cfg, device=device, freeze_model=freeze_weights)
        if no_detector:
            self.detector = None
            print('no detector is used')
        else:
            if body_detector == 'rcnn':
                self.detector = detectors.FasterRCNN(device=device)
            elif body_detector == 'keypoint':
                self.detector = detectors.KeypointRCNN(device=device)
            else:
                print('no detector is used')

        if freeze_weights and not self.detector is None:
            # self.Encoder, self.Regressor, self.Moderator, self.Extractor
            for param in self.detector.model.parameters():
                param.requires_grad = False
    
    def forward(self, x_reshaped, x_reshaped_resized, draw_bbox = False, gt_bbox = None, logger = None):
        batch = {}
        with torch.no_grad():
            batch['image_hd'] = x_reshaped
            
            bs, c, h_1, w_1 = x_reshaped_resized.size()
            bs, c, hd_h, hd_w = x_reshaped.size()
            scale_x = hd_w / w_1
            slice_y = hd_h / h_1
            
            # print('temp_img, min, max', temp_img.min(), temp_img.max())
            # cv2.imwrite('temp_img.jpg', (temp_img*255).astype(np.uint8))
            # print('bbox', bbox) # [104.55945  46.06325 168.01297 255.5114 ]
            if not gt_bbox is None:
                if not isinstance(gt_bbox, torch.Tensor):
                    gt_bbox = torch.tensor(gt_bbox, device=self.device) #gt_bbox shape [4]
                bboxes_new = gt_bbox.unsqueeze(0).expand(bs, -1)
                if draw_bbox:
                    bbox = bboxes_new[0].cpu().numpy()
                    temp_img = x_reshaped[0:1].cpu().numpy().transpose(0, 2, 3, 1).squeeze()
                    # print('temp_img', temp_img.shape, temp_img.dtype)
                    # print('temp_img', temp_img.min(), temp_img.max())
                    temp_img = (temp_img*255).astype(np.uint8)[:,:,::-1]                 
                    # draw bbox on the temp_img
                    # print('temp_img', temp_img.shape, temp_img.dtype) # (256, 256, 3) uint8
                    # print('temp_img', temp_img.min(), temp_img.max()) # 0 255
                    # print('bbox', bbox) #  [ 97.484825  82.27155  207.8292   252.12723 ]
                    temp_img = cv2.rectangle(temp_img.copy(), (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    cv2.imwrite('temp_img.jpg', temp_img)


            else:
                # print('x_reshaped_resized', x_reshaped_resized.size(), x_reshaped_resized.min(), x_reshaped_resized.max())
                bboxes = self.detector.run_batch(x_reshaped_resized)     
                # print('bboxes', bboxes)                   
                if isinstance(bboxes, np.ndarray):
                    bboxes = torch.tensor(bboxes, device=self.device)
                # print('1 bboxes', bboxes)
                if draw_bbox:
                    bbox = bboxes[0].cpu().numpy()
                    temp_img = x_reshaped_resized[0:1].cpu().numpy().transpose(0, 2, 3, 1).squeeze()
                    # print('temp_img', temp_img.shape, temp_img.dtype)
                    # print('temp_img', temp_img.min(), temp_img.max())
                    temp_img = (temp_img*255).astype(np.uint8)[:,:,::-1]                 
                    # cv2.imwrite('temp_img.jpg', temp_img)
                    # draw bbox on the temp_img
                    # print('temp_img', temp_img.shape, temp_img.dtype) # (256, 256, 3) uint8
                    # print('temp_img', temp_img.min(), temp_img.max()) # 0 255
                    # print('bbox', bbox) #  [ 97.484825  82.27155  207.8292   252.12723 ]
                    temp_img = cv2.rectangle(temp_img.copy(), (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    cv2.imwrite('temp_img.jpg', temp_img)

                    # #draw bbox on image for the last frame
                    # temp_img_last = x_reshaped[-1:].cpu().numpy().transpose(0, 2, 3, 1).squeeze()
                    # bbox = bboxes[-1].cpu().numpy()
                    # temp_img_last = (temp_img_last*255).astype(np.uint8)[:,:,::-1]
                    # cv2.imwrite('temp_img_last_0.jpg', temp_img_last)
                    # temp_img_last = cv2.rectangle(temp_img_last.copy(), (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]), (0, 255, 0), 2))
                    # cv2.imwrite('temp_img_last.jpg', temp_img_last)


                # print('bboxes.size()', bboxes.size())  # torch.Size([bs*n_frames, 4])
                bboxes_new = bboxes * torch.tensor([scale_x, slice_y, scale_x, slice_y], device=self.device)
                
            # print('bboxes_new.size()', bboxes_new.size())  # torch.Size([bs*n_frames, 4])
            # print('2 bboxes', bboxes)
            # print('bboxes_new', bboxes_new)
            # if not gt_bbox is None:
            #     if not isinstance(gt_bbox, torch.Tensor):
            #         gt_bbox = torch.tensor(gt_bbox, device=self.device)
            #     invalid_bbox_mask = (bboxes_new == torch.tensor([0, 0, 0, 0], device=self.device)).all(dim=1)
            #     bboxes_new = torch.where(invalid_bbox_mask.unsqueeze(1), gt_bbox.expand_as(bboxes_new), bboxes_new)

            if not logger is None:
                logger.info(f'bboxes_new: {bboxes_new.shape}, {bboxes_new.cpu().numpy()[0]}')
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
    
        return joints_3d
    
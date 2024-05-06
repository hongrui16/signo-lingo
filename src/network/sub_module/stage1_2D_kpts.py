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
from matplotlib import pyplot as plt

class Compute_2D_kpts(nn.Module):
    """2D Keypoints Encoder for 2D keypoints part of model."""
    def __init__(self, freeze_weights=True, load_pretrained = True, device='cuda'):
        self.num_2d_keypoints = 17
        self.rcnn_keypoints_model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=load_pretrained)
        self.rcnn_keypoints_model.eval()
        self.rcnn_keypoints_model.to(device)

        if freeze_weights:
            self.freeze_weights()

    def freeze_weights(self):        
        ## freeze keypoint RCNN all parameters
        for param in self.rcnn_keypoints_model.parameters():
            param.requires_grad = False

    def forward(self, x_reshaped_resized, plot_keypoints=False):
        assert x_reshaped_resized.ndim == 4, f"Input tensor must have 4 dimensions, but has {x_reshaped_resized.ndim}"
        
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

                

            if plot_keypoints:
                temp_keypoints = keypoints_tensors[0].cpu().numpy()
                # print('temp_keypoints', temp_keypoints.shape, temp_keypoints.dtype)
                temp_img = (x_reshaped_resized[0].cpu().numpy().transpose(1, 2, 0)*255).astype(np.uint8)
                # print(temp_img.shape, temp_img.dtype, temp_img.min(), temp_img.max())
                fig, ax = plt.subplots()
                ax.imshow(temp_img)  # 显示图像

                # 在每个关键点位置绘制一个圆
                for (x, y, v) in temp_keypoints.tolist():
                    if v > 0:
                        circle = plt.Circle((x, y), 5, color='green', fill=True)
                        ax.add_patch(circle)

                ax.set_axis_off()  # 关闭坐标轴
                plt.savefig("output.png", bbox_inches='tight', pad_inches=0)  # 保存图像
                plt.close(fig)  # 关闭图形，释放资源

                
                # draw keypoints on image

            keypoints_tensors = torch.stack(keypoints_tensors)
            return keypoints_tensors
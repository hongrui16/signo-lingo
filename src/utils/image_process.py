import os, sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import scipy
from PIL import Image

def crop_resize_image(image, bbox, scale, target_size):
    """
    根据边界框（bbox），缩放因子（scale）和目标尺寸（target_size）裁剪并调整图像尺寸。
    参数:
    - image: PIL.Image对象，原始图像。
    - bbox: 边界框，格式为(left, top, right, bottom)。
    - scale: 缩放因子，用于确定裁剪的正方形边长。
    - target_size: 要调整到的新尺寸（宽度和高度相同）。
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        raise ValueError('image must be a PIL.Image or numpy.ndarray object')
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    
    # 计算边界框的中心
    center_x = left + width / 2.0
    center_y = top + height / 2.0
    
    # 找出最长边并计算新的裁剪边长
    long_edge = max(width, height) * scale
    half_edge = long_edge / 2.0
    
    # 计算新的边界框
    new_left = int(center_x - half_edge)
    new_top = int(center_y - half_edge)
    new_right = int(center_x + half_edge)
    new_bottom = int(center_y + half_edge)
    
    # 裁剪图像
    cropped_image = image.crop((new_left, new_top, new_right, new_bottom))
    
    # 调整图像尺寸
    resized_image = cropped_image.resize((target_size, target_size), Image.ANTIALIAS)
    
    return np.array(resized_image)



def crop_resize_image_batch(image, bbox, scale, target_size):
    bs, c, h, w = image.shape
    new_bbox = bbox.clone()  # 避免修改原始bbox

    # 缩放bbox
    bbox_width = new_bbox[:, 2] - new_bbox[:, 0]
    bbox_height = new_bbox[:, 3] - new_bbox[:, 1]
    new_bbox[:, 0] -= (scale - 1) * bbox_width / 2
    new_bbox[:, 2] += (scale - 1) * bbox_width / 2
    new_bbox[:, 1] -= (scale - 1) * bbox_height / 2
    new_bbox[:, 3] += (scale - 1) * bbox_height / 2
    
    # 确保bbox不越界
    new_bbox = torch.clamp(new_bbox, 0, max(w-1, h-1))

    # 计算新的长边
    bbox_width = new_bbox[:, 2] - new_bbox[:, 0]
    bbox_height = new_bbox[:, 3] - new_bbox[:, 1]
    max_side = torch.max(bbox_width, bbox_height)

    # 为每个bbox计算填充
    left_pad = (max_side - bbox_width) / 2
    right_pad = max_side - bbox_width - left_pad
    top_pad = (max_side - bbox_height) / 2
    bottom_pad = max_side - bbox_height - top_pad
    
    # 初始化输出张量
    cropped_images = []
    
    for i in range(bs):
        # 裁剪图片
        img_cropped = image[i:i+1, :, int(new_bbox[i, 1]):int(new_bbox[i, 3]), int(new_bbox[i, 0]):int(new_bbox[i, 2])]
        # 填充图片
        img_padded = F.pad(img_cropped, (int(left_pad[i]), int(right_pad[i]), int(top_pad[i]), int(bottom_pad[i])))
        
        # 调整大小到目标尺寸
        img_resized = F.interpolate(img_padded, size=(target_size, target_size), mode='bilinear', align_corners=False)
        cropped_images.append(img_resized)
    cropped_images = torch.cat(cropped_images, dim=0)
    return cropped_images

if __name__ == '__main__':
    # 假设图片和边界框数据
    image = torch.rand(3, 300, 300)  # 300x300的图片
    bbox = torch.tensor([50, 50, 250, 250], dtype=torch.float)
    images = torch.rand(5, 3, 300, 300)  # 5张300x300的图片
    bboxes = torch.tensor([[50, 50, 250, 250], [30, 30, 200, 200], [60, 60, 240, 240], [20, 20, 220, 220], [40, 40, 230, 230]], dtype=torch.float)
    scale = 1.1
    target_size = 224

    # 调用函数
    # output_images = crop_resize_image(image, bbox, scale, target_size)
    # print(output_images.shape)  # 应输出(5, 224, 224, 3)
    output_images = crop_resize_image_batch(images, bboxes, scale, target_size)
    print(output_images.shape)  # 应输出(5, 3, 224, 224)

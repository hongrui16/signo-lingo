# matplotlib
import matplotlib.pyplot as plt

# numpy
import numpy as np
import torch.nn.functional as F
import csv

# torch
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import transforms
from torchsummary import summary
from PIL import Image
# misc
import time
from datetime import datetime
# from torchinfo import summary
import cv2

# preprocessing 
import av
import pandas as pd
import matplotlib.pyplot as plt

# logging
import os
import logging
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

import sys
import io

import argparse


from network.sub_module.stage1_3D_kpts import Compute_3D_kpts


def pad_resize(frame, hd_size = 720, bbox=None):
    h, w, _ = frame.shape
    pad_x = pad_y = 0
    if h > w:
        pad_x = (h - w) // 2
        frame = cv2.copyMakeBorder(frame, 0, 0, pad_x, pad_x, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    elif w > h:
        pad_y = (w - h) // 2
        frame = cv2.copyMakeBorder(frame, pad_y, pad_y, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    frame = cv2.resize(frame, (hd_size, hd_size))

        # Update bbox
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        scale_x = hd_size / (w + 2 * pad_x)
        scale_y = hd_size / (h + 2 * pad_y)
        
        # Adjust bbox coordinates to account for padding
        x_min_new = (x_min + pad_x) * scale_x
        y_min_new = (y_min + pad_y) * scale_y
        x_max_new = (x_max + pad_x) * scale_x
        y_max_new = (y_max + pad_y) * scale_y

        # Update bbox to the scaled and padded values
        bbox = [x_min_new, y_min_new, x_max_new, y_max_new]
        bbox = np.array(bbox).astype(int)

    return frame, bbox


def preprocess_frame_tensor(x_reshaped, input_size=256):

    # print(f'x_reshaped min: {x_reshaped.min()}, max: {x_reshaped.max()}') #min: 0.0, max: 1.0
    # resize image to input size
    batch_size, C, H, W = x_reshaped.size()
    if H != input_size or W != input_size:
        x_reshaped_resized = F.interpolate(x_reshaped, size=(input_size, input_size), mode='bilinear')
        # print(f'x_reshaped_resized min: {x_reshaped_resized.min()}, max: {x_reshaped_resized.max()}') #min: 0.0, max: 1.0
        # print(x_reshaped_resized.size())  # torch.Size([batch_size*n_frames, 3, 256, 256])
        return x_reshaped_resized
    else:
        return x_reshaped

def main_WLASL(args):
    log_dir =  args.log_dir
    log_dir = os.path.join(log_dir, "{}_{:%Y-%m-%d_%H-%M-%S}".format('3D_KPT_runtime', datetime.now()))
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "info.log")
    
    # Log to file & tensorboard writer
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
    logger = logging.getLogger('signo-lingo')
    logging.info(f"Logging to {log_dir}")

        # check if GPU is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info("Using GPU")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")




    video_root_dir = '/scratch/pfayyazs/datasets/WLASL2000'
    data_root_dir = '/scratch/rhong5/dataset/signLanguage/WLASL/WLASL300'
    keypoint_save_dir = '/scratch/rhong5/dataset/signLanguage/WLASL/3D_kpts'
    os.makedirs(keypoint_save_dir, exist_ok=True)
    
    # if 'WLASL' in video_root_dir:
    #     no_detector = True
    # else:
    #     no_detector = False
    no_detector = False
    net_3d_kps = Compute_3D_kpts(freeze_weights=True, load_pretrained = True, device=device, no_detector = no_detector)
    net_3d_kps.to(device)
    net_3d_kps.eval()

    splits = ['train', 'val', 'test']
    for split in splits:
        anno_json_file = os.path.join(data_root_dir, f'{split}.json')
        ## load json file
        with open(anno_json_file) as f:
            video_info_dict = pd.read_json(f)
        video_names = video_info_dict.keys()
        for idx, video_id in enumerate(video_names):
            # if not str(video_id) == '12338':
            #     continue
            v_info_dict = video_info_dict[video_id]
            frame_end = v_info_dict['frame_end']
            frame_start = v_info_dict['frame_start']
            bbox = v_info_dict['bbox']
            # print('bbox', bbox)
            bbox = np.array(bbox).astype(int).tolist()
            ## load video

            ## right alligen video_id, 5 bits, 0 padded left
            video_id = str(video_id).zfill(5)
            video_path = os.path.join(video_root_dir, f'{video_id}.mp4')
            logger.info(f'Processing video: {video_path}')
            logger.info(f'Processing {split}: {idx}/{len(video_names)}, {video_path}')

            # open video file with opencv
            cap = cv2.VideoCapture(video_path)

            # if faided to open video file, continue to the next video
            if not cap.isOpened():
                logger.info(f"Failed to open video file {video_path}")
                continue
            # get total number of frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)


            # Determine the range to read frames based on frame_end value
            if frame_end == -1:
                frame_end = total_frames  # Read till the end of the video

            # Adjust frame_start and frame_end to zero-based index for OpenCV
            frame_start -= 1
            if frame_end != total_frames:
                frame_end -= 1


            # get each frame
            frames = []
            ## read a frame:
            max_edge = max(height, width)
            max_edge = int(max_edge)
            if frame_end - frame_start > 90:
                frame_list = np.linspace(frame_start, frame_end, 90, endpoint=False, dtype=int)
            else:
                frame_list = np.linspace(frame_start, frame_end, frame_end - frame_start, endpoint=False, dtype=int)
            for i in frame_list:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    logger.info(f"Failed to read frame at position {i} of {video_path}.")
                    continue
                frame, new_bbox = pad_resize(frame, hd_size = max_edge, bbox = bbox)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32)
                frame /= 255.0
                frame = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).float().to(device)
                frames.append(frame)
            cap.release()
            # print('after pad, bbox', new_bbox)
            # logger.info(f'bbox: {new_bbox.tolist()}')
            if len(frames) == 0:
                logger.info(f"No frames read from {video_path}")
                continue
            frames = torch.cat(frames, dim=0)
            resized_frames = preprocess_frame_tensor(frames)
            with torch.no_grad():
                joints_3d = net_3d_kps(frames, resized_frames, draw_bbox = args.debug, logger = None)  # torch.Size([bs, 17, 3])
                # print('joints_3d.size()', joints_3d.size())  # torch.Size([bs, 17, 3])
                # plotly_save_point_cloud(joints_3d[0].cpu().numpy())
                keypoints = joints_3d.cpu().numpy().squeeze()

            ## save keypoints
            save_path = os.path.join(keypoint_save_dir, f'{video_id}.npy')
            logger.info(f'Saving keypoints to {save_path}')
            if os.path.exists(save_path):
                os.remove(save_path)
            # save_path = os.path.join('./', f'{video_id}.npy')
            np.save(save_path, keypoints)
            del frames, resized_frames, joints_3d, frame
            # break

def maiN_AUTSL(args):
    log_dir =  args.log_dir
    log_dir = os.path.join(log_dir, "{}_{:%Y-%m-%d_%H-%M-%S}".format('3D_KPT_runtime', datetime.now()))
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "info.log")
    
    # Log to file & tensorboard writer
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
    logger = logging.getLogger('signo-lingo')
    logging.info(f"Logging to {log_dir}")

        # check if GPU is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info("Using GPU")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")




    # video_root_dir = '/scratch/rhong5/dataset/signLanguage/AUTSL'
    data_root_dir = '/scratch/rhong5/dataset/signLanguage/AUTSL'
    keypoint_root_dir = os.path.join(data_root_dir, '3D_kpts')
    splits = ['train', 'val', 'test']
    for split in splits:
        kpts_save_dir = os.path.join(keypoint_root_dir, split)
        os.makedirs(kpts_save_dir, exist_ok=True)
    
    
    net_3d_kps = Compute_3D_kpts(freeze_weights=True, load_pretrained = True, device=device)
    net_3d_kps.to(device)
    net_3d_kps.eval()

    splits = ['train', 'val', 'test']
    for split in splits:
        kpts_save_dir = os.path.join(keypoint_root_dir, split)
        anno_csv_file = os.path.join(data_root_dir, f'{split}_labels.csv')
        ## load csv file
        file_infos = []
        with open(anno_csv_file, newline='', encoding='utf-8') as csvfile:
            # 创建一个CSV读取器
            reader = csv.reader(csvfile)
            # 逐行读取CSV文件
            for row in reader:
                # 如果行不为空
                if row:
                    # 获取文件名和ID，并添加到列表中
                    filename = row[0]
                    id = row[1]
                    file_infos.append((filename, id))
        video_dir = os.path.join(data_root_dir, split)

        for idx, (video_name, label_id) in enumerate(file_infos):                                
            video_path = os.path.join(video_dir, f'{video_name}_color.mp4')
            logger.info(f'Processing {split}: {idx}/{len(file_infos)}, {video_path}')
            if not os.path.exists(video_path):
                continue
            # open video file with opencv
            cap = cv2.VideoCapture(video_path)

            # if faided to open video file, continue to the next video
            if not cap.isOpened():
                logger.info(f"Failed to open video file {video_path}")
                continue
            # get total number of frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_start = 0
            frame_end = total_frames  # Read till the end of the video

            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

            max_edge = max(height, width)
            max_edge = int(max_edge)
            # get each frame
            frames = []
            if frame_end - frame_start > 90:
                frame_list = np.linspace(frame_start, frame_end, 90, endpoint=False, dtype=int)
            else:
                frame_list = np.linspace(frame_start, frame_end, frame_end - frame_start, endpoint=False, dtype=int)
            for i in frame_list:
                if i > 5 and args.debug:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    logger.info(f"Failed to read frame at position {i} of {video_path}.")
                    continue
                frame, new_bbox = pad_resize(frame, hd_size=max_edge)                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32)
                frame /= 255.0
                frame = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).float().to(device)
                frames.append(frame)
            cap.release()
            # print('after pad, bbox', new_bbox)
            # logger.info(f'bbox: {new_bbox.tolist()}')
            if len(frames) == 0:
                logger.info(f"No frames read from {video_path}")
                continue
            frames = torch.cat(frames, dim=0)
            resized_frames = preprocess_frame_tensor(frames)
            with torch.no_grad():
                joints_3d = net_3d_kps(frames, resized_frames, draw_bbox = args.debug, logger = None)  # torch.Size([bs, 17, 3])
                # print('joints_3d.size()', joints_3d.size())  # torch.Size([bs, 17, 3])
                # plotly_save_point_cloud(joints_3d[0].cpu().numpy())
                keypoints = joints_3d.cpu().numpy().squeeze()

            ## save keypoints
            save_path = os.path.join(kpts_save_dir, f'{video_name}_{label_id}.npy')
            logger.info(f'Saving keypoints to {save_path}')
            if os.path.exists(save_path):
                os.remove(save_path)
            # save_path = os.path.join('./', f'{video_id}.npy')
            np.save(save_path, keypoints)
            del frames, resized_frames, joints_3d, frame
            # break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model on AUTSL dataset')
    parser.add_argument('--log_dir', type=str, default='logs', help='directory to save logs')
    parser.add_argument('--dataset_name', type=str, default=None, help='directory to save logs')
    parser.add_argument('--debug', action='store_true', help='debug mode')

    args = parser.parse_args()
    if args.dataset_name == 'AUTSL':
        maiN_AUTSL(args)
    elif args.dataset_name == 'WLASL':
        main_WLASL(args)
    else:
        print('Please provide the dataset name')
        sys.exit(1)


'''
salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=30 --gres=gpu:A100.80gb:1 --mem=80gb -t 0-24:00:00
salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=30 --gres=gpu:A100.40gb:1 --mem=50gb -t 0-24:00:00
salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=15 --gres=gpu:3g.40gb:1 --mem=50gb -t 0-24:00:00
salloc -p contrib-gpuq -q gpu --nodes=1 --ntasks-per-node=15 --gres=gpu:3g.40gb:1 --mem=50gb -t 0-24:00:00
salloc -p contrib-gpuq -q gpu --nodes=1 --ntasks-per-node=15 --gres=gpu:3g.40gb:1 --mem=50gb -t 0-1:00:00

'''
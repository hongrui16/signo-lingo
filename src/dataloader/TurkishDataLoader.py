# matplotlib
import matplotlib.pyplot as plt

# numpy
import numpy as np

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

import sys
import io

import argparse




def extract_frames(vid_path, transforms=None, frames_cap=None): 
    """Extract and transform video frames

    Parameters:
    vid_path (str): path to video file
    frames_cap (int): number of frames to extract, evenly spaced
    transforms (torchvision.transforms, optional): transformations to apply to frame

    Returns:
    list of numpy.array: vid_arr

    """
    vid_arr = []
    try:
        container = av.open(vid_path)
    
        stream = container.streams.video[0]
        n_frames = stream.frames
        if frames_cap:
            remainder = n_frames % frames_cap
            interval = n_frames // frames_cap
            take_frame_idx = 0
            if interval < 1:
                raise ValueError(f"video with path '{vid_path}' is too short, please make sure that video has >={frames_cap} frames")
        for frame_no, frame in enumerate(container.decode(stream)):
            if frames_cap and frame_no != take_frame_idx:
                continue
            img = frame.to_image()
            if transforms:
                img = transforms(img)
            vid_arr.append(np.array(img))
            if frames_cap:
                if remainder > 0:
                    take_frame_idx += 1
                    remainder -= 1
                take_frame_idx += interval

        return vid_arr
    except av.error.InvalidDataError as e:
        # print("Invalid data found when processing input:", e)
        return None


def extract_frames_by_cv2(vid_path, transforms=None, frames_cap=None, hd_size=720):
    """Extract and transform video frames using OpenCV.

    Parameters:
    vid_path (str): path to video file
    frames_cap (int): number of frames to extract, evenly spaced
    transforms (callable, optional): function to apply transformations to frame

    Returns:
    list of numpy.array: vid_arr

    """
    vid_arr = []
    cap = cv2.VideoCapture(vid_path)
    
    if not cap.isOpened():
        # print(f"Failed to open video file {vid_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frames_cap:
        interval = max(1, total_frames // frames_cap)
        frames_to_capture = [i * interval for i in range(frames_cap)]
    
    frame_idx = 0
    valid_frame_cnt = 0
    shapes = []
    while True:
        ret, frame = cap.read()
        if not ret:
            success = False
            break
        
        # find the longer side of the frame and then pad the shorter side to make it square, then resize to hd_size
        h, w, _ = frame.shape
        if h > w:
            pad = (h - w) // 2
            frame = cv2.copyMakeBorder(frame, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        elif w > h:
            pad = (w - h) // 2
            frame = cv2.copyMakeBorder(frame, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        frame = cv2.resize(frame, (hd_size, hd_size))

        if frames_cap:
            if frame_idx in frames_to_capture:
                valid_frame_cnt += 1
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if transforms:
                    frame = Image.fromarray(frame)
                    frame = transforms(frame)
                    frame = frame.numpy()
                
                vid_arr.append(frame)
                shapes.append(frame.shape)

                if valid_frame_cnt >= frames_cap:
                    success = True
                    break
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if transforms:
                frame = Image.fromarray(frame)
                frame = transforms(frame)
                frame = frame.numpy()
            vid_arr.append(frame)
            shapes.append(frame.shape)

        frame_idx += 1

    # print("total frames:", len(vid_arr), vid_arr[0].shape)
    if len(vid_arr) < frames_cap:
        success = False

    cap.release()
    if success:
        return vid_arr
    else:
        return None
    



"""
# Custom Dataset
"""

class Turkish_Dataset(Dataset):
    """Custom dataset class for AUTSL Dataset."""
    
    def __init__(self, df, data_dir, n_classes, file_ext=".mp4", transforms=None, frames_cap=None, hd_size=720):
        
        self.df = df
        self.frames_cap = frames_cap
        self.transforms = transforms
        self.file_ext = file_ext
        self.data_dir = data_dir
        self.n_classes = n_classes
        self.num = len(self.df)
        self.hd_size = hd_size
    
    def __getitem__(self, index):

        while True:            
            vid_name = self.df.iloc[index, 0]
            vid_label = self.df.iloc[index, 1]
            
            vid_color = f"{self.data_dir}/{vid_name}_color{self.file_ext}"

            # get videos
            # rgb_arr = extract_frames(vid_color, transforms=self.transforms, frames_cap=self.frames_cap)
            rgb_arr = extract_frames_by_cv2(vid_color, frames_cap=self.frames_cap, hd_size=self.hd_size)
            
            if not rgb_arr is None:
                break
            else:
                index = np.random.randint(0, self.num)

        vid_arr = np.array(rgb_arr)
        # print(f'1 vid_arr min: {vid_arr.min()}, vid_arr max: {vid_arr.max()}') # min: 0.0, vid_arr max: 255.0
        vid_arr = vid_arr / 255
        # print(f'2 vid_arr min: {vid_arr.min()}, vid_arr max: {vid_arr.max()}') # min: 0.0, vid_arr max: 1.0
        
        # create one-hot-encoding for label
        label = np.zeros(self.n_classes)
        label[vid_label] = 1
        
        # convert arr to tensors
        vid_arr = torch.from_numpy(vid_arr).float()
        vid_arr = vid_arr.permute(0, 3, 1, 2)
        label = torch.from_numpy(label).long().argmax()
        # shapes = torch.from_numpy(np.array(shapes)).int()
        
        # return masked video array and label
        return vid_arr, label
    
    def __len__(self):
        return len(self.df)




'''
salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=30 --gres=gpu:A100.80gb:1 --mem=80gb -t 0-24:00:00
salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=30 --gres=gpu:A100.40gb:1 --mem=50gb -t 0-24:00:00
salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=15 --gres=gpu:3g.40gb:1 --mem=50gb -t 0-24:00:00
salloc -p contrib-gpuq -q gpu --nodes=1 --ntasks-per-node=15 --gres=gpu:3g.40gb:1 --mem=50gb -t 0-24:00:00

'''
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

import sys, os
sys.path.append('../..')

# sys.path.append(os.path.abspath(os.path.join(os.getcwd, '../..')))

from utils.kpts_trainsform import rotate_keypoints_sequence, scale_keypoints_sequence, translate_keypoints_sequence, add_noise_to_keypoints_sequence



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

class AUTSL_Frame_Dataset(Dataset):
    """Custom dataset class for AUTSL Dataset."""
    
    def __init__(self, data_dir = "/scratch/rhong5/dataset/signLanguage/AUTSL/", split = 'train', file_ext=".mp4", 
                 transforms=None, frames_cap=None, hd_size=720):
        
        self.frames_cap = frames_cap
        self.transforms = transforms
        self.file_ext = file_ext
        self.hd_size = hd_size
        self.split = split

        self.video_dir = os.path.join(data_dir, split)
        
        ### check if the data directory is empty
        if not os.listdir(self.video_dir):
            raise ValueError(f"Data directory '{self.video_dir}' is empty")

        ## check if the data directory has no video files
        if not any(file.endswith(file_ext) for file in os.listdir(self.video_dir)):
            raise ValueError(f"Data directory '{self.video_dir}' has no video files")

        # All labels
        filtered_data = "/home/rhong5/research_pro/hand_modeling_pro/signo-lingo/src/data/"
        self.label_df = pd.read_csv(f'{filtered_data}/{split}.csv', header=None)

        # Total label + turkish to english translation

        total_label = pd.read_csv(f'{filtered_data}/filtered_ClassId.csv')
        self.n_classes = len(total_label['ClassId'].unique())

        ### Processing test: 3741/3742, /scratch/rhong5/dataset/signLanguage/AUTSL/test/signer30_sample662_color.mp4
        ### Saving keypoints to /scratch/rhong5/dataset/signLanguage/AUTSL/3D_kpts/test/signer30_sample662_63.npy
        
    def __len__(self):
        return len(self.label_df)


    def __getitem__(self, index):

        while True:            
            vid_name = self.label_df.iloc[index, 0]
            vid_label = self.label_df.iloc[index, 1]
            
            video_filepath = f"{self.video_dir}/{vid_name}_color{self.file_ext}"

            # get videos
            # rgb_arr = extract_frames(vid_color, transforms=self.transforms, frames_cap=self.frames_cap)
            rgb_arr = extract_frames_by_cv2(video_filepath, frames_cap=self.frames_cap, hd_size=self.hd_size)
            
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






class AUTSL_3D_KPTS_Dataset(Dataset):
    def __init__(self, data_root_dir = '/scratch/rhong5/dataset/signLanguage/AUTSL', dataset_name = 'AUTSL', 
                 file_ext = 'npy', split = 'train', transforms=None, frames_cap=60):
        
        self.data_root_dir = data_root_dir
        self.keypoint_dir = os.path.join(data_root_dir, '3D_kpts', split)
        ### check if the data directory is empty
        if not os.listdir(self.keypoint_dir):
            raise ValueError(f"Key point directory '{self.keypoint_dir}' is empty")

        ## check if the data directory has no video files
        if not any(file.endswith(file_ext) for file in os.listdir(self.keypoint_dir)):
            raise ValueError(f"Key point directory '{self.keypoint_dir}' has no {file_ext} files")

        self.npy_filenames = [f.split('.')[0] for f in os.listdir(self.keypoint_dir) if f.endswith('.npy')]
        self.transforms = transforms
        self.frames_cap = frames_cap
        anno_csv_file = os.path.join(self.data_root_dir, f'{split}_labels.csv')
        ## csv file have n rows and 2 columns, first column is video name, second column is class label, get all class labels
        self.df = pd.read_csv(anno_csv_file, header=None)
        ## get all class labels
        self.labels = self.df.iloc[:, 1].unique().tolist()
        # print(f"1 labels: {self.labels}")
        # ## convert class labels to a int list
        # self.labels = [int(label) for label in self.labels]
        # print(f"2 labels: {self.labels}")

        # sort the int list
        self.labels.sort() ## from 0 to 225
        # print(f"3 labels: {self.labels}")

        self.n_classes = len(self.labels)
        print(f"total unique label: {self.n_classes}")

    def __len__(self):
        return len(self.npy_filenames)

    def __getitem__(self, idx):
        kpts_filepath = os.path.join(self.keypoint_dir, self.npy_filenames[idx]+'.npy')
        keypoint_array = np.load(kpts_filepath)

        keypoint_length = len(keypoint_array)
        stride = keypoint_length // self.frames_cap

        if keypoint_length < self.frames_cap:
            # 重复最后一组关键点直到填满 frames_cap
            repeat_count = self.frames_cap - keypoint_length
            last_frame_repeats = np.repeat(keypoint_array[-1:], repeat_count, axis=0)
            keypoint_array = np.concatenate([keypoint_array, last_frame_repeats], axis=0)
        elif keypoint_length > self.frames_cap:
            if stride >= 2:
                # 商大于等于2，进行均匀采样
                keypoint_array = keypoint_array[::stride][:self.frames_cap]
            else:
                # 商小于2，进行随机采样
                indices = np.random.choice(keypoint_length, self.frames_cap, replace=False)
                keypoint_array = keypoint_array[np.sort(indices)]

        keypoint_array = scale_keypoints_sequence(keypoint_array)
        keypoint_array = rotate_keypoints_sequence(keypoint_array)
        keypoint_array = translate_keypoints_sequence(keypoint_array)
        keypoint_array = add_noise_to_keypoints_sequence(keypoint_array)
        
        
        label_index = self.npy_filenames[idx].split('_')[-1]
        label_index = int(label_index)

        label = np.zeros(self.n_classes)
        label[label_index] = 1
        label = torch.from_numpy(label).long().argmax()

        keypoint_array = keypoint_array.reshape(self.frames_cap, -1)
        keypoint_array = torch.from_numpy(keypoint_array).float()

        return keypoint_array, label
    



if __name__ == '__main__':

    dataset = AUTSL_3D_KPTS_Dataset(split='train', frames_cap=60)
    print(len(dataset))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    for i, (inputs, labels) in enumerate(dataloader):
        print(inputs.shape, labels.shape)
        if i == 0:
            break

    dataset = AUTSL_Frame_Dataset(split='train', frames_cap=30)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    for i, (inputs, labels) in enumerate(dataloader):
        print(inputs.shape, labels.shape)
        if i == 0:
            break


'''
salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=30 --gres=gpu:A100.80gb:1 --mem=80gb -t 0-24:00:00
salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=30 --gres=gpu:A100.40gb:1 --mem=50gb -t 0-24:00:00
salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=15 --gres=gpu:3g.40gb:1 --mem=50gb -t 0-24:00:00
salloc -p contrib-gpuq -q gpu --nodes=1 --ntasks-per-node=15 --gres=gpu:3g.40gb:1 --mem=50gb -t 0-24:00:00

'''
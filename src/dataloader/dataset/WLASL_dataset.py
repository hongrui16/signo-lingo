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

from utils.kpts_trainsform import rotate_keypoints_sequence, scale_keypoints_sequence, translate_keypoints_sequence, add_noise_to_keypoints_sequence

class WLASL_3D_KPTS_Dataset(Dataset):
    def __init__(self, data_root_dir = '/scratch/rhong5/dataset/signLanguage/WLASL', dataset_name = 'WLASL100', 
                 file_ext = 'npy', split = 'train', transforms=None, frames_cap=60):
        
        self.data_root_dir = data_root_dir
        self.keypoint_dir = os.path.join(data_root_dir, '3D_kpts')
        ### check if the data directory is empty
        if not os.listdir(self.keypoint_dir):
            raise ValueError(f"Key point directory '{self.keypoint_dir}' is empty")

        ## check if the data directory has no video files
        if not any(file.endswith(file_ext) for file in os.listdir(self.keypoint_dir)):
            raise ValueError(f"Key point directory '{self.keypoint_dir}' has no {file_ext} files")

        npy_filenames = [f.split('.')[0] for f in os.listdir(self.keypoint_dir) if f.endswith('.npy')]
        self.transforms = transforms
        self.frames_cap = frames_cap
        anno_json_file = os.path.join(self.data_root_dir, dataset_name, f'{split}.json')
        ## load json file
        with open(anno_json_file) as f:
            self.video_info_dict = pd.read_json(f)
        self.video_names = self.video_info_dict.keys()


        ## filter out the video names that are not in the keypoint directory
        self.video_names = [video_id for video_id in self.video_names if str(video_id).zfill(5) in npy_filenames]
        ## load labels list from txt file
        label_set_filepath = os.path.join(self.data_root_dir, dataset_name, 'label_set.txt')
        with open(label_set_filepath, 'r') as f:
            self.labels = f.read().splitlines()
        self.n_classes = len(self.labels)

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_id = self.video_names[idx]
        video_path = os.path.join(self.keypoint_dir, f'{str(video_id).zfill(5)}.npy')
        keypoint_array = np.load(video_path)

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
        
        gloss = self.video_info_dict[video_id]['gloss']
        label_index = self.labels.index(gloss)
        label = np.zeros(self.n_classes)
        label[label_index] = 1
        label = torch.from_numpy(label).long().argmax()

        keypoint_array = keypoint_array.reshape(self.frames_cap, -1)
        keypoint_array = torch.from_numpy(keypoint_array).float()

        return keypoint_array, label


class WLASL_Frame_Dataset(Dataset):
    def __init__(self, data_root_dir = '/scratch/rhong5/dataset/signLanguage/WLASL', dataset_name = 'WLASL100', 
                 file_ext = 'mp4', split = 'train', transforms=None, frames_cap=60, hd_size=720):
        
        self.video_root_dir = '/scratch/pfayyazs/datasets/WLASL2000'
        self.data_root_dir = data_root_dir
        
        ### check if the data directory is empty
        if not os.listdir(self.video_root_dir):
            raise ValueError(f"Data directory '{self.video_root_dir}' is empty")

        ## check if the data directory has no video files
        if not any(file.endswith(file_ext) for file in os.listdir(self.video_root_dir)):
            raise ValueError(f"Data directory '{self.video_root_dir}' has no video files")
        
        all_video_names = [f.split('.')[0] for f in os.listdir(self.video_root_dir) if f.endswith('.mp4')]
        self.transforms = transforms
        self.frames_cap = frames_cap
        self.hd_size = hd_size
        anno_json_file = os.path.join(self.data_root_dir, dataset_name, f'{split}.json')
        ## load json file
        with open(anno_json_file) as f:
            self.video_info_dict = pd.read_json(f)
        self.video_names = self.video_info_dict.keys()


        ## filter out the video names that are not in the keypoint directory
        self.video_names = [video_id for video_id in self.video_names if str(video_id).zfill(5) in all_video_names]
        ## load labels list from txt file
        label_set_filepath = os.path.join(self.data_root_dir, dataset_name, 'label_set.txt')
        with open(label_set_filepath, 'r') as f:
            self.labels = f.read().splitlines()
        self.n_classes = len(self.labels)



        self.data_root_dir = data_root_dir
        self.transforms = transforms
        self.frames_cap = frames_cap
        self.hd_size = hd_size
        anno_json_file = os.path.join(self.data_root_dir, dataset_name, f'{split}.json')
        ## load json file
        with open(anno_json_file) as f:
            self.video_info_dict = pd.read_json(f)
        self.video_names = self.video_info_dict.keys()

        ## load labels list from txt file
        label_set_filepath = os.path.join(self.data_root_dir, dataset_name, 'label_set.txt')
        with open(label_set_filepath, 'r') as f:
            self.labels = f.read().splitlines()

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_id = self.video_names[idx]
        video_path = os.path.join(self.keypoint_dir, f'{str(video_id).zfill(5)}.npy')
        keypoint_array = np.load(video_path)

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


        gloss = self.video_info_dict[video_id]['gloss']
        label_index = self.labels.index(gloss)
        label = np.zeros(self.n_classes)
        label[label_index] = 1
        label = torch.from_numpy(label).long().argmax()

        keypoint_array = keypoint_array.reshape(self.frames_cap, -1)
        keypoint_array = torch.from_numpy(keypoint_array).float()

        return keypoint_array, label


class WLASL_2D_KPTS_Dataset(Dataset):
    def __init__(self, data_root_dir = '/scratch/rhong5/dataset/signLanguage/WLASL', dataset_name = 'WLASL100', 
                 file_ext = 'mp4', split = 'train', transforms=None, frames_cap=60, hd_size=720):
        
        self.video_root_dir = '/scratch/pfayyazs/datasets/WLASL2000'
        self.data_root_dir = data_root_dir
        
        ### check if the data directory is empty
        if not os.listdir(self.video_root_dir):
            raise ValueError(f"Data directory '{self.video_root_dir}' is empty")

        ## check if the data directory has no video files
        if not any(file.endswith(file_ext) for file in os.listdir(self.video_root_dir)):
            raise ValueError(f"Data directory '{self.video_root_dir}' has no video files")
        
        all_video_names = [f.split('.')[0] for f in os.listdir(self.video_root_dir) if f.endswith('.mp4')]
        self.transforms = transforms
        self.frames_cap = frames_cap
        self.hd_size = hd_size
        anno_json_file = os.path.join(self.data_root_dir, dataset_name, f'{split}.json')
        ## load json file
        with open(anno_json_file) as f:
            self.video_info_dict = pd.read_json(f)
        self.video_names = self.video_info_dict.keys()


        ## filter out the video names that are not in the keypoint directory
        self.video_names = [video_id for video_id in self.video_names if str(video_id).zfill(5) in all_video_names]
        ## load labels list from txt file
        label_set_filepath = os.path.join(self.data_root_dir, dataset_name, 'label_set.txt')
        with open(label_set_filepath, 'r') as f:
            self.labels = f.read().splitlines()
        self.n_classes = len(self.labels)



        self.data_root_dir = data_root_dir
        self.transforms = transforms
        self.frames_cap = frames_cap
        self.hd_size = hd_size
        anno_json_file = os.path.join(self.data_root_dir, dataset_name, f'{split}.json')
        ## load json file
        with open(anno_json_file) as f:
            self.video_info_dict = pd.read_json(f)
        self.video_names = self.video_info_dict.keys()

        ## load labels list from txt file
        label_set_filepath = os.path.join(self.data_root_dir, dataset_name, 'label_set.txt')
        with open(label_set_filepath, 'r') as f:
            self.labels = f.read().splitlines()

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_id = self.video_names[idx]
        video_path = os.path.join(self.keypoint_dir, f'{str(video_id).zfill(5)}.npy')
        keypoint_array = np.load(video_path)

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


        gloss = self.video_info_dict[video_id]['gloss']
        label_index = self.labels.index(gloss)
        label = np.zeros(self.n_classes)
        label[label_index] = 1
        label = torch.from_numpy(label).long().argmax()

        keypoint_array = keypoint_array.reshape(self.frames_cap, -1)
        keypoint_array = torch.from_numpy(keypoint_array).float()

        return keypoint_array, label
    

if __name__ == '__main__':

    dataset_name = 'WLASL100'
    dataset_name = 'WLASL300'
    dataset = WLASL_3D_KPTS_Dataset(dataset_name = dataset_name)
    print(len(dataset))
    print(dataset.labels)
    # print(dataset.video_info_dict)
    # print(dataset.video_names)
    # print(dataset.video_info_dict[dataset.video_names[0]])

    ## print keypoint_array, label shape
    keypoint_array, label = dataset[0]
    print(keypoint_array.shape, label.shape)

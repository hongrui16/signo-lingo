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
import matplotlib.pyplot as plt

# preprocessing 
import av
import pandas as pd
import os

from dataloader.dataset.AUTSL_dataset import AUTSL_Frame_Dataset, AUTSL_3D_KPTS_Dataset
from dataloader.dataset.WLASL_dataset import WLASL_3D_KPTS_Dataset, WLASL_Frame_Dataset, WLASL_2D_KPTS_Dataset




def create_dataloader(log_dir, logging, args):
    """ Create dataset for training, validation, and testing. """
    hd_size = args.hd_size
    n_frames = args.n_frames


    transforms_compose = None
    train_set = None
    test_set = None
    val_set = None

    logging.info(f"Creating Dataset")
    if args.dataset_name == 'AUTSL':
        if args.load_frame:
        
            train_set = AUTSL_Frame_Dataset(split = 'train',transforms=transforms_compose, frames_cap=n_frames, hd_size=hd_size)
            logging.info(f"shape of first array: {train_set[0][0].shape}")

            # show image but clip rbg values
            img_np_arr = train_set[0][0][0].numpy()
            # print(f'img_np_arr min: {img_np_arr.min()}, img_np_arr max: {img_np_arr.max()}') # min: 0.0, img_np_arr max: 0.003921568859368563
            img_np_arr -= img_np_arr.min() 
            img_np_arr /= img_np_arr.max()

            plt.imshow(img_np_arr.transpose(1, 2, 0))
            # plt.show()
            plot_path = os.path.join(log_dir, "train_30_0.png")
            plt.savefig(plot_path)

            # create test dataset
            test_set = AUTSL_Frame_Dataset(split = 'test',transforms=transforms_compose, frames_cap=n_frames, hd_size=hd_size)

            # create val dataset
            val_set = AUTSL_Frame_Dataset(split = 'val',transforms=transforms_compose, frames_cap=n_frames, hd_size=hd_size)
            n_classes = train_set.n_classes

        elif args.load_3D_kpts:
            n_frames = 60
            train_set = AUTSL_3D_KPTS_Dataset(dataset_name = args.dataset_name, split = 'train', frames_cap=n_frames)
            test_set = AUTSL_3D_KPTS_Dataset(dataset_name = args.dataset_name, split = 'test', frames_cap=n_frames)
            val_set = AUTSL_3D_KPTS_Dataset(dataset_name = args.dataset_name, split = 'val', frames_cap=n_frames)
            n_classes = train_set.n_classes

        else:
            raise ValueError("Please select the correct dataset type: 3D_kpts, 2D_kpts, or frame_only")
        
    elif 'WLASL' in args.dataset_name:
        assert args.dataset_name in ['WLASL100', 'WLASL300', 'WLASL1000','WLASL2000'], "Please provide correct dataset name"
        # n_frames = 30
        n_frames = 60
        if args.load_3D_kpts:
            train_set = WLASL_3D_KPTS_Dataset( dataset_name = args.dataset_name, split = 'train', frames_cap=n_frames)
            test_set = WLASL_3D_KPTS_Dataset( dataset_name = args.dataset_name, split = 'test', frames_cap=n_frames)
            val_set = WLASL_3D_KPTS_Dataset( dataset_name = args.dataset_name, split = 'val', frames_cap=n_frames)
            n_classes = train_set.n_classes
        elif args.load_2D_kpts:
            train_set = WLASL_2D_KPTS_Dataset( dataset_name = args.dataset_name, split = 'train', frames_cap=n_frames)
            test_set = WLASL_2D_KPTS_Dataset( dataset_name = args.dataset_name, split = 'test', frames_cap=n_frames)
            val_set = WLASL_2D_KPTS_Dataset( dataset_name = args.dataset_name, split = 'val', frames_cap=n_frames)
            n_classes = train_set.n_classes
        elif args.load_frame:
            train_set = WLASL_Frame_Dataset( dataset_name = args.dataset_name, split = 'train', frames_cap=n_frames)
            test_set = WLASL_Frame_Dataset( dataset_name = args.dataset_name, split = 'test', frames_cap=n_frames)
            val_set = WLASL_Frame_Dataset( dataset_name = args.dataset_name, split = 'val', frames_cap=n_frames)
            n_classes = train_set.n_classes
        else:
            raise ValueError("Please select the correct dataset type: 3D_kpts, 2D_kpts, or frame_only")

    else:
        raise ValueError("Dataset not supported")
    

    logging.info(f"total unique label: {n_classes}")

    # create all dataloaders
    bs_train = args.batch_size
    bs_test = args.batch_size
    bs_val = args.batch_size
    if not train_set is None:
        train_loader = DataLoader(train_set, batch_size = bs_train, shuffle = True, drop_last=True, num_workers = args.num_workers)
    else:
        train_loader = None
    
    if not test_set is None:
        test_loader = DataLoader(test_set, batch_size = bs_test, shuffle = False, num_workers = args.num_workers)
    else:
        test_loader = None

    if not val_set is None:
        val_loader = DataLoader(val_set, batch_size = bs_val, shuffle = False, num_workers = args.num_workers)
    else:
        val_loader = None

    return train_loader, test_loader, val_loader, n_classes
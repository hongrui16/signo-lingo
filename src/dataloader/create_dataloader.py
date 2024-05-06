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

from dataloader.dataset.AUTSL_dataset import AUTSLDataset
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
        data_dir = "/scratch/rhong5/dataset/signLanguage/AUTSL/"
        train_dir = f'{data_dir}/train'
        val_dir = f'{data_dir}/val'
        test_dir = f'{data_dir}/test'

        # All labels
        filtered_data = "/home/rhong5/research_pro/hand_modeling_pro/signo-lingo/src/data/"
        train_label_df = pd.read_csv(f'{filtered_data}/train.csv', header=None)
        test_label_df = pd.read_csv(f'{filtered_data}/test.csv', header=None)
        val_label_df = pd.read_csv(f'{filtered_data}/val.csv', header=None)

        # Total label + turkish to english translation

        total_label = pd.read_csv(f'{filtered_data}/filtered_ClassId.csv')
        n_classes = len(total_label['ClassId'].unique())
        logging.info(f"total unique label: {n_classes}")

        train_set = AUTSLDataset(train_label_df, train_dir, n_classes, transforms=transforms_compose, frames_cap=n_frames, hd_size=hd_size)
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
        test_set = AUTSLDataset(test_label_df, test_dir, n_classes, transforms=transforms_compose, frames_cap=n_frames, hd_size=hd_size)
        # print("shape of first array", ld_test[0][0].shape)


        # create val dataset
        val_set = AUTSLDataset(val_label_df, val_dir, n_classes, transforms=transforms_compose, frames_cap=n_frames, hd_size=hd_size)
        # print("shape of first array", ld_val[0][0].shape)
        
    elif 'WLASL' in args.dataset_name:
        assert args.dataset_name in ['WLASL100', 'WLASL300', 'WLASL1000','WLASL2000'], "Please provide correct dataset name"
        # n_frames = 30
        n_frames = 60
        if args.load_3D_kpts_only:
            train_set = WLASL_3D_KPTS_Dataset( dataset_name = args.dataset_name, split = 'train', frames_cap=n_frames)
            test_set = WLASL_3D_KPTS_Dataset( dataset_name = args.dataset_name, split = 'test', frames_cap=n_frames)
            val_set = WLASL_3D_KPTS_Dataset( dataset_name = args.dataset_name, split = 'val', frames_cap=n_frames)
            n_classes = train_set.n_classes
        elif args.load_2D_kpts_only:
            train_set = WLASL_2D_KPTS_Dataset( dataset_name = args.dataset_name, split = 'train', frames_cap=n_frames)
            test_set = WLASL_2D_KPTS_Dataset( dataset_name = args.dataset_name, split = 'test', frames_cap=n_frames)
            val_set = WLASL_2D_KPTS_Dataset( dataset_name = args.dataset_name, split = 'val', frames_cap=n_frames)
            n_classes = train_set.n_classes
        elif args.load_frame_only:
            train_set = WLASL_Frame_Dataset( dataset_name = args.dataset_name, split = 'train', frames_cap=n_frames)
            test_set = WLASL_Frame_Dataset( dataset_name = args.dataset_name, split = 'test', frames_cap=n_frames)
            val_set = WLASL_Frame_Dataset( dataset_name = args.dataset_name, split = 'val', frames_cap=n_frames)
            n_classes = train_set.n_classes
        else:
            raise ValueError("Please select the correct dataset type: 3D_kpts, 2D_kpts, or frame_only")

    else:
        raise ValueError("Dataset not supported")

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
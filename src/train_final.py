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
from torch.utils.tensorboard import SummaryWriter

import sys
import io

import argparse


# helper classes and functions
from workers import trainval, test
from network.network import SLR_network
from dataloader.TurkishDataLoader import Turkish_Dataset


class CaptureOutput:
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._capture = io.StringIO()
        return self

    def __exit__(self, type, value, traceback):
        sys.stdout = self._stdout
        self.captured = self._capture.getvalue()






def main(args):
    log_dir =  args.log_dir
    # use_2d_kps = args.use_2d_kps
    # use_3d_kps = args.use_3d_kps
    # use_img_feats = args.use_img_feats
    resume = args.resume
    fine_tune = args.fine_tune
    debug = args.debug
    batch_size = args.batch_size
    num_workers = args.num_workers
    max_epoch = args.max_epoch

    input_size = args.input_size
    hd_size = args.hd_size
    model_name = args.model_name
    n_frames = args.n_frames
    encoder_types = args.encoder_types.split(',')
    decoder_types = args.decoder_types.split(',')
    

    log_dir = os.path.join(log_dir, "{}_{:%Y-%m-%d_%H-%M-%S}".format(model_name, datetime.now()))
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "info.log")
    
    # Log to file & tensorboard writer
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
    logger = logging.getLogger('signo-lingo')
    logging.info(f"Logging to {log_dir}")

    ## print all the args into log file
    kwargs = vars(args)
    for key, value in kwargs.items():
        logger.info(f"{key}: {value}")
    
    
    writer = SummaryWriter(log_dir)

    data_dir = "/scratch/rhong5/dataset/signLanguage/AUTSL/"
    train_dir = f'{data_dir}/train'
    val_dir = f'{data_dir}/val'
    test_dir = f'{data_dir}/test'

    # All labels
    filtered_data = "./data"
    train_label_df = pd.read_csv(f'{filtered_data}/train.csv', header=None)
    test_label_df = pd.read_csv(f'{filtered_data}/test.csv', header=None)
    val_label_df = pd.read_csv(f'{filtered_data}/val.csv', header=None)

    # Total label + turkish to english translation

    total_label = pd.read_csv(f'{filtered_data}/filtered_ClassId.csv')
    n_classes = len(total_label['ClassId'].unique())
    print("total unique label:", n_classes)


    # create train dataset
    transforms_compose = transforms.Compose([
                                         #transforms.Resize(256), 
                                         transforms.ToTensor(),
                                        #  transforms.Normalize(mean=[0.5], std=[0.5]),
                                         ])
    transforms_compose = None
    
    ld_train = Turkish_Dataset(train_label_df, train_dir, n_classes, transforms=transforms_compose, frames_cap=n_frames, hd_size=hd_size)
    print("shape of first array", ld_train[0][0].shape)

    # show image but clip rbg values
    img_np_arr = ld_train[0][0][0].numpy()
    print(f'img_np_arr min: {img_np_arr.min()}, img_np_arr max: {img_np_arr.max()}') # min: 0.0, img_np_arr max: 0.003921568859368563

    img_np_arr -= img_np_arr.min() 
    img_np_arr /= img_np_arr.max()

    plt.imshow(img_np_arr.transpose(1, 2, 0))
    # plt.show()
    plot_path = os.path.join(log_dir, "train_30_0.png")
    plt.savefig(plot_path)

    # create test dataset
    ld_test = Turkish_Dataset(test_label_df, test_dir, n_classes, transforms=transforms_compose, frames_cap=n_frames, hd_size=hd_size)
    # print("shape of first array", ld_test[0][0].shape)

    # show image but clip rbg values
    img_np_arr = ld_test[0][0][0].numpy()
    img_np_arr -= img_np_arr.min() 
    img_np_arr /= img_np_arr.max()
    plt.imshow(img_np_arr.transpose(1, 2, 0))
    # plt.show()
    plot_path = os.path.join(log_dir, "test_30_0.png")
    plt.savefig(plot_path)


    # create val dataset
    ld_val = Turkish_Dataset(val_label_df, val_dir, n_classes, transforms=transforms_compose, frames_cap=n_frames, hd_size=hd_size)
    # print("shape of first array", ld_val[0][0].shape)

    # show image but clip rbg values
    img_np_arr = ld_val[0][0][0].numpy()
    img_np_arr -= img_np_arr.min() 
    img_np_arr /= img_np_arr.max()
    plt.imshow(img_np_arr.transpose(1, 2, 0))
    # plt.show()
    plot_path = os.path.join(log_dir, "val_30_0.png")
    plt.savefig(plot_path)

    # 
    """
    # Custom Dataloader
    """

    # create all dataloaders
    bs_train = batch_size
    bs_test = batch_size
    bs_val = batch_size
    train_loader = DataLoader(ld_train, batch_size = bs_train, shuffle = True, drop_last=True, num_workers = num_workers)
    test_loader = DataLoader(ld_test, batch_size = bs_test, shuffle = False, num_workers = num_workers)
    val_loader = DataLoader(ld_val, batch_size = bs_val, shuffle = False, num_workers = num_workers)


    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info("Using GPU")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
      

    """
    # Train Final Model
    """
    if model_name == 'SLR_network':
        model = SLR_network(n_classes, 
                        device = device,
                        encoder_types = encoder_types, #['2D_kpts', '3D_kpts', 'img_feats'],
                        decoder_types = decoder_types, #['TransformerDecoder', 'LSTM'],
                        input_size=input_size)
    else:
        raise ValueError("Model name not supported")
    
    with CaptureOutput() as capturer:
        summary(model.cuda(), input_size=(n_frames, 3, 256, 256))
    logger.info(capturer.captured)

    # if multiple GPUs are available, wrap model with DataParallel 
    if torch.cuda.device_count() > 1:
        logger.info("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    # hyperparams
    optimizer_lr = 1e-4
    logger.info(f"******************* Training {model_name} *******************")
    # train model
    trainval(model, 
        train_loader, 
        val_loader, 
        max_epoch, 
        logger,
        writer,
        save_weight_dir=log_dir, 
        device=device, 
        patience=10, 
        optimizer_lr=optimizer_lr, 
        use_scheduler=True,
        resume=resume,
        fine_tune=fine_tune,
        debug=debug,
        test_loader=test_loader)

    # test model
    logger.info("Testing Model".center(60, '#'))
    test_loss, test_acc = test(model, test_loader, log_dir, device, debug=debug, max_epoch=max_epoch)

    logger.info(f"Test Loss: {test_loss:4f}, Test Accuracy: {test_acc*100:2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model on AUTSL dataset')
    parser.add_argument('--log_dir', type=str, default='logs', help='directory to save logs')
    # parser.add_argument('--use_2d_kps', action='store_true', help='use 2D keypoints')
    # parser.add_argument('--use_3d_kps', action='store_true', help='use 3D keypoints')
    # parser.add_argument('--use_img_feats', action='store_true', help='use image features')

    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume training')
    parser.add_argument('--fine_tune', action='store_true', help='fine tune model')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--num_workers', type=int, default=6, help='number of workers for dataloader')
    parser.add_argument('--max_epoch', type=int, default=150, help='maximum number of epochs')
    parser.add_argument('--input_size', type=int, default=256, help='input size of image')
    parser.add_argument('--hd_size', type=int, default=720, help='high resolution size')
    parser.add_argument('--model_name', type=str, default='SLR_network', help='model name')
    parser.add_argument('--n_frames', type=int, default=30, help='number of frames in a sequence for training')
    parser.add_argument('--encoder_types', type=str, default='3D_kpts', help='encoder types, separated by comma')
    parser.add_argument('--decoder_types', type=str, default='TransformerDecoder', help='decoder types, separated by comma')

    args = parser.parse_args()
    main(args)


'''
salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=30 --gres=gpu:A100.80gb:1 --mem=80gb -t 0-24:00:00
salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=30 --gres=gpu:A100.40gb:1 --mem=50gb -t 0-24:00:00
salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=15 --gres=gpu:3g.40gb:1 --mem=50gb -t 0-24:00:00
salloc -p contrib-gpuq -q gpu --nodes=1 --ntasks-per-node=15 --gres=gpu:3g.40gb:1 --mem=50gb -t 0-24:00:00

'''
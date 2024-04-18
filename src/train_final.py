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
from dev_model.model_fns import trainval, test
from dev_model.models import CNN_LSTM, VGG_LSTM



class CaptureOutput:
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._capture = io.StringIO()
        return self

    def __exit__(self, type, value, traceback):
        sys.stdout = self._stdout
        self.captured = self._capture.getvalue()


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


def extract_frames_by_cv2(vid_path, transforms=None, frames_cap=None):
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
    while True:
        ret, frame = cap.read()
        if not ret:
            success = False
            break
        

        
        if frames_cap:
            if frame_idx in frames_to_capture:
                valid_frame_cnt += 1
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if transforms:
                    frame = Image.fromarray(frame)
                    frame = transforms(frame)
                    frame = frame.numpy()
                vid_arr.append(frame)

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

        frame_idx += 1

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
    
    def __init__(self, df, data_dir, n_classes, file_ext=".mp4", transforms=None, frames_cap=None):
        
        self.df = df
        self.frames_cap = frames_cap
        self.transforms = transforms
        self.file_ext = file_ext
        self.data_dir = data_dir
        self.n_classes = n_classes
        self.num = len(self.df)
    
    def __getitem__(self, index):

        while True:            
            vid_name = self.df.iloc[index, 0]
            vid_label = self.df.iloc[index, 1]
            
            vid_color = f"{self.data_dir}/{vid_name}_color{self.file_ext}"

            # get videos
            # rgb_arr = extract_frames(vid_color, transforms=self.transforms, frames_cap=self.frames_cap)
            rgb_arr = extract_frames_by_cv2(vid_color, transforms=self.transforms, frames_cap=self.frames_cap)
            
            if not rgb_arr is None:
                break
            else:
                index = np.random.randint(0, self.num)

        vid_arr = np.array(rgb_arr)

        vid_arr = vid_arr / 255
        
        # create one-hot-encoding for label
        label = np.zeros(self.n_classes)
        label[vid_label] = 1
        
        # convert arr to tensors
        vid_arr = torch.from_numpy(vid_arr).float()
        label = torch.from_numpy(label).long().argmax()
        
        # return masked video array and label
        return vid_arr, label
    
    def __len__(self):
        return len(self.df)


transforms_compose = transforms.Compose([transforms.Resize(256), 
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5], std=[0.5])])



def main(args):
    log_dir =  args.log_dir
    use_2d_kps = args.use_2d_kps
    use_3d_kps = args.use_3d_kps
    resume = args.resume
    fine_tune = args.fine_tune
    debug = args.debug
    batch_size = args.batch_size
    num_workers = args.num_workers
    max_epoch = args.max_epoch
    use_img_feats = args.use_img_feats

    log_dir = os.path.join(log_dir, "cnnlstm_{:%Y-%m-%d_%H-%M-%S}".format(datetime.now()))
    os.makedirs(log_dir, exist_ok=True)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log_path = os.path.join(log_dir, "info.log")
    
    # Log to file & tensorboard writer
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
    logger = logging.getLogger('signo-lingo')
    logging.info(f"Logging to {log_path}")

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
    n_frames = 30
    ld_train = Turkish_Dataset(train_label_df, train_dir, n_classes, transforms=transforms_compose, frames_cap=n_frames)
    print("shape of first array", ld_train[0][0].shape)

    # show image but clip rbg values
    img_np_arr = ld_train[0][0][0].numpy()
    img_np_arr -= img_np_arr.min() 
    img_np_arr /= img_np_arr.max()

    plt.imshow(img_np_arr.transpose(1, 2, 0))
    # plt.show()
    plot_path = os.path.join(log_dir, "train_30_0.png")
    plt.savefig(plot_path)

    # create test dataset
    ld_test = Turkish_Dataset(test_label_df, test_dir, n_classes, transforms=transforms_compose, frames_cap=n_frames)
    print("shape of first array", ld_test[0][0].shape)

    # show image but clip rbg values
    img_np_arr = ld_test[0][0][0].numpy()
    img_np_arr -= img_np_arr.min() 
    img_np_arr /= img_np_arr.max()
    plt.imshow(img_np_arr.transpose(1, 2, 0))
    # plt.show()
    plot_path = os.path.join(log_dir, "test_30_0.png")
    plt.savefig(plot_path)


    # create val dataset
    ld_val = Turkish_Dataset(val_label_df, val_dir, n_classes, transforms=transforms_compose, frames_cap=n_frames)
    print("shape of first array", ld_val[0][0].shape)

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

    """
    # Train Final Model
    """

    # create model
    model = CNN_LSTM(n_classes, 
                    latent_size=512, 
                    n_cnn_layers=6, 
                    n_rnn_layers=1, 
                    n_rnn_hidden_dim=512, 
                    cnn_bn=True, 
                    bidirectional=True, 
                    dropout_rate=0.8, 
                    attention=True,
                    use_2d_kps=use_2d_kps,
                    use_3d_kps=use_3d_kps,
                    use_img_feats=use_img_feats)
    
    with CaptureOutput() as capturer:
        summary(model.cuda(), input_size=(30, 3, 256, 256))
    logger.info(capturer.captured)

    # if multiple GPUs are available, wrap model with DataParallel 
    if torch.cuda.device_count() > 1:
        logger.info("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    # hyperparams
    optimizer_lr = 1e-5

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
    parser = argparse.ArgumentParser(description='Train CNN-LSTM model on AUTSL dataset')
    parser.add_argument('--log_dir', type=str, default='logs', help='directory to save logs')
    parser.add_argument('--use_2d_kps', action='store_true', help='use 2D keypoints')
    parser.add_argument('--use_3d_kps', action='store_true', help='use 3D keypoints')
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume training')
    parser.add_argument('--fine_tune', action='store_true', help='fine tune model')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--num_workers', type=int, default=6, help='number of workers for dataloader')
    parser.add_argument('--max_epoch', type=int, default=500, help='maximum number of epochs')
    parser.add_argument('--use_img_feats', action='store_true', help='use image features')

    args = parser.parse_args()
    main(args)


'''
salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=30 --gres=gpu:A100.80gb:1 --mem=80gb -t 0-24:00:00
salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=30 --gres=gpu:A100.40gb:1 --mem=50gb -t 0-24:00:00
salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=15 --gres=gpu:3g.40gb:1 --mem=50gb -t 0-24:00:00
salloc -p contrib-gpuq -q gpu --nodes=1 --ntasks-per-node=15 --gres=gpu:3g.40gb:1 --mem=50gb -t 0-24:00:00

'''
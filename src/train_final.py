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
import torch.optim as optim

import sys
import io
import argparse

from network.network import SLR_network
from dataloader.create_dataloader import create_dataloader

from workers import save_checkpoint, train_epoch, val_epoch, test_runtime

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
    resume = args.resume
    fine_tune = args.fine_tune
    debug = args.debug
    max_epoch = args.max_epoch

    input_size = args.input_size
    n_frames = args.n_frames
    dataset_name = args.dataset_name
    model_name = args.model_name
    encoder_types = args.encoder_types.split(',')
    decoder_types = args.decoder_types.split(',')
    

    log_dir = os.path.join(log_dir, "{}_{}_{:%Y-%m-%d_%H-%M-%S}".format(model_name, dataset_name, datetime.now()))
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "info.log")
    
    # Log to file & tensorboard writer
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
    logger = logging.getLogger('signo-lingo')
    logging.info(f"Logging to {log_dir}")

    ## print all the args into log file
    logging.info(f"<<<<<<<<<<<<<<<<***************hyperparameters***********************************************")
    kwargs = vars(args)
    for key, value in kwargs.items():
        logger.info(f"{key}: {value}")
    logging.info(f"*******************************hyperparameters*******************************>>>>>>>>>>>>>>>>>")
    
    writer = SummaryWriter(log_dir)

    """
    # Custom Dataloader
    """
    logging.info("Creating Dataloaders")
    train_loader, test_loader, val_loader, n_classes = create_dataloader(log_dir, logger, args)
    

    # check if GPU is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info("Using GPU")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
      
    ### create model
    if model_name == 'SLR_network':
        model = SLR_network(n_classes, 
                        device = device,
                        encoder_types = encoder_types, #['2D_kpts', '3D_kpts', 'img_feats'],
                        decoder_types = decoder_types, #['TransformerDecoder', 'LSTM'],
                        trans_num_layers = args.trans_num_layers,
                        input_size=input_size,
                        args=args)
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
    patience=10
    use_scheduler = True

    """Train function for model."""
    
    # move model to device specified
    model.to(device)


    # initialise loss function and optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience//3)


    start_epoch = 0
    best_val_acc = float("-inf")
    best_epoch = None

    if resume is not None:
        # load model from checkpoint
        checkpoint = torch.load(resume)
        if not fine_tune:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint['val_acc']
            if 'best_epoch' in checkpoint:
                best_epoch = checkpoint['best_epoch']
    

    logger.info(f"******************* Training {model_name} *******************")
    logger.info(f"Start Epoch: {start_epoch}")


    # start training
    for epoch in range(start_epoch, max_epoch+1):
        #logger.info(f"Epoch {epoch}")
        
        # train the model
        train_loss, train_acc = train_epoch(epoch, max_epoch, model, criterion, optimizer, train_loader, device, debug=debug)

        # write train loss and acc to logger
        writer.add_scalars('Loss', {'train': train_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc}, epoch)
        logger.info("Average Training Loss of Epoch {}: {:.6f} | Acc: {:.2f}%".format(epoch, train_loss, train_acc*100))

        # validate the model
        val_loss, val_acc = val_epoch(epoch, max_epoch, model, criterion, val_loader, device, split='val', debug=debug)

        # write val loss and acc to logger
        writer.add_scalars('Loss', {'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'val': val_acc}, epoch)
        logger.info("Average Validation Loss of Epoch {}: {:.6f} | Acc: {:.2f}%".format(epoch, val_loss, val_acc*100))

        
        # step scheduler
        if use_scheduler:
            scheduler.step(val_loss)
        
        if not test_loader is None and epoch % 20 == 0:
            test_loss, test_acc = val_epoch(epoch, max_epoch, model, criterion, test_loader, device, split='test', debug=debug)
            logger.info("Average Test Loss of Epoch {}: {:.6f} | Acc: {:.2f}%".format(epoch, test_loss, test_acc*100))

        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'best_epoch': best_epoch

        }
        # save model or checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_checkpoint(checkpoint_dict, log_dir, is_best=True)
            logger.info("Best Epoch: {} | Validation Loss: {:.6f} | Acc: {:.2f}%".format(best_epoch, val_loss, val_acc*100))
        else:
            save_checkpoint(checkpoint_dict, log_dir)
        
        # update and check early stopper
        #if early_stopper.stop(val_loss):
        #    logger.info("Model has overfit, early stopping...")
        #    break
        
        logger.info("")
        if debug:
            break
        
    logger.info("Training Finished".center(60, '#'))
    logger.info("")
    logger.info("")

    # test model
    logger.info("Testing Model".center(60, '#'))
    test_loss, test_acc = test_runtime(model, test_loader, log_dir, device, debug=debug, max_epoch=max_epoch)

    logger.info(f"Test Loss: {test_loss:4f}, Test Accuracy: {test_acc*100:2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model on AUTSL dataset')
    parser.add_argument('--log_dir', type=str, default='logs', help='directory to save logs')
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume training')
    parser.add_argument('--fine_tune', action='store_true', help='fine tune model')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--num_workers', type=int, default=6, help='number of workers for dataloader')
    parser.add_argument('--max_epoch', type=int, default=250, help='maximum number of epochs')
    parser.add_argument('--input_size', type=int, default=256, help='input size of image')
    parser.add_argument('--hd_size', type=int, default=720, help='high resolution size')
    parser.add_argument('--model_name', type=str, default='SLR_network', help='model name')
    parser.add_argument('--n_frames', type=int, default=30, help='number of frames in a sequence for training')
    parser.add_argument('--encoder_types', type=str, default='3D_kpts', help='encoder types, separated by comma')
    parser.add_argument('--decoder_types', type=str, default='TransformerDecoder', help='decoder types, separated by comma')
    parser.add_argument('--trans_num_layers', type=int, default=4, help='number of layers in transformer encoder or decoder')
    parser.add_argument('--dataset_name', type=str, default='AUTSL', help='dataset name')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--load_3D_kpts', action='store_true', help='load 3D keypoints data as input to model')
    parser.add_argument('--load_2D_kpts', action='store_true', help='load 2D keypoints data as input to model')
    parser.add_argument('--load_frame', action='store_true', help='load frame as input to model')

    args = parser.parse_args()
    main(args)


'''
salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=30 --gres=gpu:A100.80gb:1 --mem=80gb -t 0-24:00:00
salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=30 --gres=gpu:A100.40gb:1 --mem=50gb -t 0-24:00:00
salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=15 --gres=gpu:3g.40gb:1 --mem=50gb -t 0-24:00:00
salloc -p contrib-gpuq -q gpu --nodes=1 --ntasks-per-node=15 --gres=gpu:3g.40gb:1 --mem=50gb -t 0-24:00:00

'''
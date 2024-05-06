from collections import OrderedDict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvBlock(nn.Module):
    """Convolution block used in CNN."""
    def __init__(self, 
                 channel_in, 
                 channel_out, 
                 activation_fn, 
                 use_batchnorm, 
                 pool:str='max_2',
                 kernel_size:int=3):
        
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, channel_out, kernel_size)
        self.conv2 = nn.Conv2d(channel_out, channel_out, kernel_size)

        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            self.batchnorm1 = nn.BatchNorm2d(channel_out, momentum=0.01)
            self.batchnorm2 = nn.BatchNorm2d(channel_out, momentum=0.01)
        
        if activation_fn == "relu":
            self.a_fn = nn.ReLU()
        elif activation_fn == "leaky_relu":
            self.a_fn = nn.LeakyReLU()
        elif activation_fn == "param_relu":
            self.a_fn = nn.PReLU()
        else:
            raise ValueError("please use a valid activation function argument ('relu'; 'leaky_relu'; 'param_relu')")

        if pool == "max_2":
            self.pool = nn.MaxPool2d(2)
        elif pool == "adap":
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif not pool:
            self.pool = pool
        else:
            raise ValueError("please use a valid pool argument ('max_2', 'adap', None)")
    
    def forward(self, x):
        out = self.conv1(x)
        if self.use_batchnorm:
            out = self.batchnorm1(out)
        out = self.a_fn(out)

        out = self.conv2(out)
        if self.use_batchnorm:
            out = self.batchnorm2(out)
        out = self.a_fn(out)
        if self.pool:
            out = self.pool(out)
        return out


class CNN_Encoder(nn.Module):
    """CNN Encoder for CNN part of model."""
    def __init__(self, 
                 channel_out = 512, 
                 n_layers = 6, 
                 intermediate_act_fn="relu", 
                 use_batchnorm=True, 
                 channel_in=3):
        
        super(CNN_Encoder, self).__init__()

        channels = [64, 128, 256, 512]
        self.channel_out = channel_out
        
        if n_layers < 1 or n_layers > len(channels)*2:
            raise ValueError(f"please use a valid int for the n_layers param (1-{len(channels)*2})")
        
        n_repeat = remainder = max(0, n_layers - len(channels))
        pointer = 0

        self.conv1 = nn.Conv2d(channel_in, channels[0], 3, stride=2)
        self.maxpool = nn.MaxPool2d(3, 2)

        layers =  OrderedDict()

        if n_layers > 1:
            layers[str(0)] = ConvBlock(channels[0], channels[0], intermediate_act_fn, use_batchnorm=use_batchnorm)

        for i in range(1, n_layers-1):
            if i % 2 == 0 and remainder > 0:
                layers[str(i)] = ConvBlock(channels[pointer], channels[pointer], intermediate_act_fn, use_batchnorm=use_batchnorm, pool=None)
                remainder -= 1
            else:
                layers[str(i)] = ConvBlock(channels[pointer], channels[min(pointer+1, len(channels)-1)], intermediate_act_fn, use_batchnorm=use_batchnorm)
                pointer += 1

        self.layers = nn.Sequential(layers)
        if n_layers < len(channels):
            conv_to_fc = channels[n_layers-1-n_repeat]
        else:
            conv_to_fc = channels[-1]
        
        self.conv2 = ConvBlock(channels[n_layers-2-n_repeat], conv_to_fc, intermediate_act_fn, use_batchnorm=use_batchnorm, pool="adap")
        
        self.fc = nn.Linear(conv_to_fc, channel_out)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.layers(out)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

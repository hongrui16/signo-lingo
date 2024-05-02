import torch
from torch import nn
import math
import numpy as np
import sys, os

# Assuming joints_3d is the input tensor reshaped as [batch_size, seq_len, features]
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, device='cpu'):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        # print('d_model:', d_model)
        even_term_count = d_model // 2
        odd_term_count = (d_model + 1) // 2  # 确保为所有奇数索引计算足够的项
        div_term = torch.exp(torch.arange(0, odd_term_count, device = device).float() * (-math.log(10000.0) / d_model))
        # print('div_term.shape:', div_term.shape)

        # Apply sin to even indices
        self.encoding[:, 0::2] = torch.sin(position * div_term[:odd_term_count])

        # Apply cos to odd indices
        self.encoding[:, 1::2] = torch.cos(position * div_term[:even_term_count])

        self.encoding = self.encoding.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        '''
        input:
            x: [seq_len, batch_size, feature_dim]
        output:
            x: [seq_len, batch_size, feature_dim]
        '''
        # print('x.shape:', x.shape)
        return x + self.encoding[:x.size(0), :]



class TransformerDecoder(nn.Module):
    def __init__(self, num_classes, d_model, nhead, num_decoder_layers = 4, dim_feedforward = 512, device = 'cpu', input_dim = None):
        super(TransformerDecoder, self).__init__()
        self.input_dim = input_dim
        if not input_dim is None:
            self.input_projection = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        self.pos_encoder = PositionalEncoding(d_model, device = device)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, device = device)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.output_layer = nn.Linear(d_model, num_classes)
        self.device = device

    def forward(self, x):
        '''
        INPUT:
            x: [batch_size, seq_len, feature_dim]
        output:
            cls: [batch_size, num_classes]
        '''
        if not self.input_dim is None:
            x = self.input_projection(x)
        x = x.transpose(0, 1)  # 将输入转换为 [seq_len, batch_size, feature_dim]
        x = self.pos_encoder(x)
        memory = torch.zeros_like(x, device=self.device)  # 初始化memory
        output = self.transformer_decoder(x, memory)  # 
        # print('output.shape:', output.shape)
        output = output[-1, :, :]  # 假设使用最后一个输出作为分类依据
        # print('output.shape:', output.shape)
        cls = self.output_layer(output)
        return cls, output

class TransformerFull(nn.Module):
    def __init__(self, num_classes, d_model, nhead = 3, num_layers = 3, device = 'cpu', input_dim = None):
        super(TransformerFull, self).__init__()
        self.input_dim = input_dim
        self.device = device
        if not input_dim is None:
            self.input_projection = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        self.pos_encoder = PositionalEncoding(d_model, device = device)

        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x):
        '''
        INPUT:
            x: [batch_size, seq_len, feature_dim]
        output:
            cls: [batch_size, num_classes]
        '''
        if not self.input_dim is None:
            x = self.input_projection(x)
        # Change the shape to [seq_len, batch_size, features]
        x = x.permute(1, 0, 2)  # Transformer expects [seq_len, batch_size, feature]
        x = self.pos_encoder(x)
        # Create a dummy target sequence for decoder
        tgt = torch.zeros_like(x, device= self.device)
        output = self.transformer(x, tgt)
        # output = output.mean(dim=0)  # Aggregate across the sequence
        output = output[-1, :, :]  # 假设使用最后一个输出作为分类依据

        cls = self.fc_out(output)
        return cls, output


class TransformerEncoderCls(nn.Module):
    def __init__(self, num_classes, d_model, nhead = 3, num_layers = 4, device = 'cpu', input_dim = None, dropout = 0.2):
        super(TransformerEncoderCls, self).__init__()
        self.input_dim = input_dim
        if not input_dim is None:
            self.input_projection = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        self.pos_encoder = PositionalEncoding(d_model, device = device)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x):
        '''
        INPUT:
            x: [batch_size, seq_len, feature_dim]
        output:
            cls: [batch_size, num_classes]
        '''
        if not self.input_dim is None:
            x = self.input_projection(x)
        # Change the shape to [seq_len, batch_size, features]
        x = x.permute(1, 0, 2)  # Transformer expects [seq_len, batch_size, feature]
        x = self.pos_encoder(x)
        # Create a dummy target sequence for decoder
        output = self.transformer_encoder(x)
        # output = output.mean(dim=0)  # Aggregate across the sequence
        output = output[-1, :, :]  # 假设使用最后一个输出作为分类依据

        cls = self.fc_out(output)
        return cls, output
    
if __name__ == '__main__':
    # Test the module
    batch_size = 1
    seq_len = 3
    feature_dim = 87*3
    # feature_dim = 4
    num_classes = 10
    pos_enc = PositionalEncoding(feature_dim)
    x = torch.zeros(seq_len, batch_size, feature_dim)
    y = pos_enc(x)
    print('y.shape:', y.shape)

    x = torch.randn(batch_size, seq_len, feature_dim)
    # print('y:', y)
    decoder = TransformerFull(num_classes, feature_dim, 3)
    sign_word = decoder(x)
    print('sign_word.shape:', sign_word.shape)
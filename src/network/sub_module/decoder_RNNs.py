from collections import OrderedDict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LSTM_Decoder(nn.Module):
    """LSTM Decoder for RNN part of model."""
    def __init__(self, 
                 embed_dim, 
                 hidden_dim, 
                 channel_out, 
                 n_layers, 
                 intermediate_act_fn="relu", 
                 bidirectional=False, 
                 attention=False,
                 device="cuda"):
        
        super(LSTM_Decoder, self).__init__()

        self.num_layers = n_layers
        self.hidden_dim = hidden_dim
        self.device = device
        self.bidirectional = bidirectional
        self.attention = attention

        self.lstm = nn.LSTM(embed_dim, 
                            self.hidden_dim, 
                            num_layers=self.num_layers, 
                            bidirectional=self.bidirectional, 
                            batch_first=True)

        if self.bidirectional:
            fc1_in = self.hidden_dim * 2
        else:
            fc1_in = self.hidden_dim
        self.fc1 =  nn.Linear(fc1_in, channel_out)

        if intermediate_act_fn == "relu":
            self.a_fn = nn.ReLU()
        elif intermediate_act_fn == "leaky_relu":
            self.a_fn = nn.LeakyReLU()
        elif intermediate_act_fn == "param_relu":
            self.a_fn = nn.PReLU()
        else:
            raise ValueError("please use a valid activation function argument ('relu'; 'leaky_relu'; 'param_relu')")
        
        if self.attention:
            self.attention_layer = nn.Linear(2 * self.hidden_dim if self.bidirectional else self.hidden_dim, 1)

    def forward(self, x):
        
        if self.bidirectional:
            h_0_size = c_0_size = self.num_layers * 2
        else:
            h_0_size = c_0_size = self.num_layers
        h = torch.zeros(h_0_size, x.size(0), self.hidden_dim).to(self.device)
        c = torch.zeros(c_0_size, x.size(0), self.hidden_dim).to(self.device)

        self.lstm.flatten_parameters()

        # Propagate input through LSTM
        out, (h, c) = self.lstm(x, (h, c)) #lstm with input, hidden, and internal state
        
        if self.attention:
            attention_w = F.softmax(self.attention_layer(out).squeeze(-1), dim=-1)
            
            out = torch.sum(attention_w.unsqueeze(-1) * out, dim=1)
        else:
            out = out[:, -1, :]
            
        out = self.fc1(out)
        
        return out, h

class GRU_Decoder(nn.Module):

    def __init__(self, 
                 embed_dim, 
                 hidden_dim, 
                 channel_out, 
                 n_layers, 
                 intermediate_act_fn="relu", 
                 bidirectional=False, 
                 device="cuda"):
        
        super(GRU_Decoder, self).__init__()

        self.num_layers = n_layers
        self.hidden_dim = hidden_dim
        self.device = device
        self.bidirectional = bidirectional

        self.gru = nn.GRU(embed_dim, 
                            self.hidden_dim, 
                            num_layers=self.num_layers, 
                            bidirectional=self.bidirectional, 
                            batch_first=True)

        if self.bidirectional:
            fc1_in = self.hidden_dim * 2
        else:
            fc1_in = self.hidden_dim
        self.fc1 =  nn.Linear(fc1_in, channel_out)

        if intermediate_act_fn == "relu":
            self.a_fn = nn.ReLU()
        elif intermediate_act_fn == "leaky_relu":
            self.a_fn = nn.LeakyReLU()
        elif intermediate_act_fn == "param_relu":
            self.a_fn = nn.PReLU()
        else:
            raise ValueError("please use a valid activation function argument ('relu'; 'leaky_relu'; 'param_relu')")

    def forward(self, x):
        
        if self.bidirectional:
            h_0_size = self.num_layers * 2
        else:
            h_0_size = self.num_layers
        h = torch.zeros(h_0_size, x.size(0), self.hidden_dim).to(self.device)

        self.gru.flatten_parameters()

        # Propagate input through LSTM
        output, h = self.gru(x, h) #lstm with input, hidden, and internal state
        
        out = output[:, -1, :]

        out = self.fc1(out)
        
        return out

class RNN_Decoder(nn.Module):

    def __init__(self, 
                 embed_dim, 
                 hidden_dim, 
                 channel_out, 
                 n_layers, 
                 intermediate_act_fn="relu", 
                 bidirectional=False, 
                 attention=False,
                 device="cuda"):
        
        super(RNN_Decoder, self).__init__()

        self.num_layers = n_layers
        self.hidden_dim = hidden_dim
        self.device = device
        self.bidirectional = bidirectional
        self.attention = attention

        self.rnn = nn.RNN(embed_dim, 
                            self.hidden_dim, 
                            num_layers=self.num_layers, 
                            bidirectional=self.bidirectional, 
                            batch_first=True)

        if self.bidirectional:
            fc1_in = self.hidden_dim * 2
        else:
            fc1_in = self.hidden_dim
        self.fc1 =  nn.Linear(fc1_in, channel_out)

        if intermediate_act_fn == "relu":
            self.a_fn = nn.ReLU()
        elif intermediate_act_fn == "leaky_relu":
            self.a_fn = nn.LeakyReLU()
        elif intermediate_act_fn == "param_relu":
            self.a_fn = nn.PReLU()
        else:
            raise ValueError("please use a valid activation function argument ('relu'; 'leaky_relu'; 'param_relu')")

    def forward(self, x):
        
        if self.bidirectional:
            h_0_size = self.num_layers * 2
        else:
            h_0_size = self.num_layers
        h = torch.zeros(h_0_size, x.size(0), self.hidden_dim).to(self.device)

        self.rnn.flatten_parameters()
        
        # Propagate input through LSTM
        out, h = self.rnn(x, h) #lstm with input, hidden, and internal state
        
        if not self.attention:
            out = out[:, -1, :]
            out = self.fc1(out)
        
        return out, h
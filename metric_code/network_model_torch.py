# -*- coding: utf-8 -*-
"""
@author: Anurag Kumar
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy 
from helper import *

class LossNet(nn.Module):
    def __init__(self, in_channels, n_layers, kernel_size, keep_prob, norm_type='sbn'):
        super().__init__()
        self.net = nn.ModuleList()
        for i in range(n_layers):
            #Increase output channels every 5 layers
            out_channels = 32 * (2 ** (i // 5))
            prev_out = 32 * (2 ** ((i-1) // 5))
            layers = nn.ModuleList()
            if i == 0:
                layer = nn.Conv2d(in_channels, 
                                  out_channels, 
                                  (1, kernel_size), 
                                  stride=(1, 2),
                                  padding=1)
            elif i == n_layers - 1:
                layer = nn.Conv2d(prev_out, 
                                  out_channels,
                                  (1, kernel_size),
                                  stride=(1, 2),
                                  padding=1)
            else:
                layer = nn.Conv2d(prev_out, 
                                  out_channels,
                                  (1, kernel_size),
                                  stride=(1, 2),
                                  padding=1)
            
            layers.append(layer)

            if norm_type == 'sbn':
                batch_norm = torch.nn.BatchNorm2d(out_channels)
            elif norm_type == 'nm':
                batch_norm = nm_torch()
            elif norm_type == 'none':
                batch_norm is None
            else:
                raise NotImplementedError
            if batch_norm is not None:
                layers.append(batch_norm)

            if i < n_layers - 1:
                dropout = nn.Dropout(1 - keep_prob)
                layers.append(dropout)

            self.net.append(layers)
        
    def forward(self, x):
        outs = []
        for layer in self.net:
            for sub_layer in layer:
                x = sub_layer(x)
            outs.append(x)
        return outs

class ClassificationHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.dense3 = nn.Linear(in_dim, 16)
        self.dense4 = nn.Linear(16, 6)
        self.dense2 = nn.Linear(6, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.dense3(x))
        out = self.relu(self.dense4(out))
        out = self.dense2(out)
        return out

                
class FeatureLossBatch(nn.Module):
    def __init__(self, n_layers, base_channels, gpu_id=None):
        super().__init__()
        self.out_channels = [base_channels * (2 ** (i // 5)) for i in range(n_layers)]
        self.weights = [nn.Parameter(torch.randn(features, 1), requires_grad=True) for features in self.out_channels]
        if gpu_id is not None:
            self.weights = [param.to(gpu_id) for param in self.weights]

    def forward(self, embeds1, embeds2):
        """
        Both embeds1 and embeeds are outputs from each layer of
        loss_net. 
        """
        loss_vec = []
        for i, (e1, e2) in enumerate(zip(embeds1, embeds2)):
            dist = e1 - e2
            dist = dist.permute(0, 1, 3, 2)
            res = self.weights[i] * dist
            loss = l1_loss_batch_torch(res)
            loss_vec.append(loss)
        return loss_vec, loss_vec


class JNDModel(nn.Module):
    def __init__(self, in_channels, n_layers, keep_prob, norm_type='sbn', gpu_id=None):
        super().__init__()
        self.loss_net = LossNet(in_channels=in_channels, 
                                n_layers=n_layers, 
                                kernel_size=3, 
                                keep_prob=keep_prob, 
                                norm_type=norm_type)

        self.classification_layer = ClassificationHead(in_dim=1, out_dim=2)

        self.feature_loss = FeatureLossBatch(n_layers=n_layers,
                                             base_channels=32,
                                             gpu_id=gpu_id)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp, ref):
        ref = self.loss_net(ref)
        inp = self.loss_net(inp)

        others, loss_sum = self.feature_loss(ref, inp)
        others = torch.stack(others)
        #print(f"others:{others.shape}")
        dist = self.sigmoid(others).reshape(-1, 1, 1)

        logits = self.classification_layer(dist).squeeze(1)
        return logits




        









        




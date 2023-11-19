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
            out_channels = 32 * (2 ** (i // 5))
            prev_out = 32 * (2 ** ((i-1) // 5))
            if i == 0:
                if norm_type == 'sbn':
                    layer = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, (kernel_size, 1), (2, 1)),
                        nn.ZeroPad2d((0, 0, 0, 1)),
                        nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU(0.2),
                        nn.Dropout(1 - keep_prob),
                    )
                if norm_type == 'nm':
                    layer = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, (kernel_size, 1), (2, 1)),
                        nn.ZeroPad2d((0, 0, 0, 1)),
                        nm_torch(out_channels),
                        nn.LeakyReLU(0.2),
                        nn.Dropout(1 - keep_prob)
                    )
                if norm_type == 'none':
                    layer = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, (kernel_size, 1), (2, 1)),
                        nn.ZeroPad2d((0, 0, 0, 1)),
                        nn.Dropout(1 - keep_prob),
                        nn.LeakyReLU(0.2),
                    )

            elif i == n_layers - 1:
                if norm_type == 'sbn':
                    layer = nn.Sequential(
                        nn.Conv2d(prev_out, out_channels, (kernel_size, 1), (2, 1)),
                        nn.ZeroPad2d((0, 0, 0, 1)),
                        nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU(0.2),
                    )
                if norm_type == 'nm':
                    layer = nn.Sequential(
                        nn.Conv2d(prev_out, out_channels, (kernel_size, 1), (2, 1)),
                        nn.ZeroPad2d((0, 0, 0, 1)),
                        nm_torch(out_channels),
                        nn.LeakyReLU(0.2),   
                    )
                if norm_type == 'none':
                    layer = nn.Sequential(
                        nn.Conv2d(prev_out, out_channels, (kernel_size, 1), (2, 1)),
                        nn.ZeroPad2d((0, 0, 0, 1)),
                        nn.LeakyReLU(0.2),
                    )
            else:
                if norm_type == 'sbn':
                    layer = nn.Sequential(
                        nn.Conv2d(prev_out, out_channels, (kernel_size, 1), (2, 1)),
                        nn.ZeroPad2d((0, 0, 0, 1)),
                        nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU(0.2),
                        nn.Dropout(1 - keep_prob),
                    )
                if norm_type == 'nm':
                    layer = nn.Sequential(
                        nn.Conv2d(prev_out, out_channels, (kernel_size, 1), (2, 1)),
                        nn.ZeroPad2d((0, 0, 0, 1)),
                        nm_torch(out_channels),
                        nn.LeakyReLU(0.2),
                        nn.Dropout(1 - keep_prob)
                    )
                if norm_type == 'none':
                    layer = nn.Sequential(
                        nn.Conv2d(prev_out, out_channels, (kernel_size, 1), (2, 1)),
                        nn.ZeroPad2d((0, 0, 0, 1)),
                        nn.LeakyReLU(0.2),
                        nn.Dropout(1 - keep_prob)
                    )
            self.net.append(layer)
        
    def forward(self, x):
        outs = []
        for layer in self.net:
            x = layer(x)
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
    def __init__(self, n_layers, base_channels, weights=False, gpu_id=None):
        super().__init__()
        self.out_channels = [base_channels * (2 ** (i // 5)) for i in range(n_layers)]
        self.out_dims = []
        for i in range(n_layers):
            if out_dim % 2 == 0: 
                out_dim = 20000 // (2 ** i)
            else:
                out_dim = 20000 // (2 ** i) + 1
            self.out_dims.append(out_dim)

        if weights:
            self.weights = [nn.Parameter(torch.randn(features, 1, out_dim), requires_grad=True) for features, out_dim in zip(self.out_channels, self.out_dims)]
            if gpu_id is not None:
                self.weights = [param.to(gpu_id) for param in self.weights]
        else:
            self.weights = None

    def forward(self, embeds1, embeds2):
        """
        Both embeds1 and embeeds are outputs from each layer of
        loss_net. 
        """
        loss_vec = []
        for i, (e1, e2) in enumerate(zip(embeds1, embeds2)):
            dist = e1 - e2
            dist = dist.permute(0, 1, 3, 2)
            print(f"dist:{dist.shape}, weight:{self.weights[i].shape}")
            if self.weights is not None:
                res = self.weights[i] * dist
            else:
                res = dist
            print(f"feature_loss:{res.shape}")
            loss = l1_loss_batch_torch(res)
            loss_vec.append(loss)
        return loss_vec


class JNDModel(nn.Module):
    def __init__(self, in_channels, n_layers, keep_prob, norm_type='sbn', gpu_id=None):
        super().__init__()
        self.loss_net = LossNet(in_channels=in_channels, 
                                n_layers=n_layers, 
                                kernel_size=3, 
                                keep_prob=keep_prob, 
                                norm_type=norm_type)

        self.classification_layer = ClassificationHead(in_dim=1, out_dim=1)

        self.feature_loss = FeatureLossBatch(n_layers=n_layers,
                                             base_channels=32,
                                             gpu_id=gpu_id,
                                             weights=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp, ref):
        ref = self.loss_net(ref)
        inp = self.loss_net(inp)

        others = self.feature_loss(ref, inp)
        dist = torch.stack(others).mean(0)
        dist = self.sigmoid(dist).reshape(-1, 1, 1)
        logits = self.classification_layer(dist)
        
        return logits




        









        




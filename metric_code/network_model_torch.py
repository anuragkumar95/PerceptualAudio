import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy 
rom helper import *

class LossNet(nn.Module):
    def __init__(self, in_channels, n_layers, kernel_size, keep_prob, norm_type='sbn'):
        super().__init__()
        self.net = nn.ModuleList()
        for i in range(n_layers):
            #Increase output channels every 5 layers
            out_channels = 32 * (2 ** (i // 5))
            prev_out = 32 * (2 ** ((i-1) // 5))
            if i == 0:
                layer = nn.Conv2d(in_channels, 
                                  out_channels, 
                                  (1, kernel_size), 
                                  stride=(1, 2),
                                  padding='same')
            elif i == n_layers - 1:
                layer = nn.Conv2d(prev_out, 
                                  out_channels,
                                  (1, kernel_size),
                                  stride=(1, 2),
                                  padding='same')
            else:
                layer = nn.Conv2d(prev_out, 
                                  out_channels,
                                  (1, kernel_size),
                                  stride=(1, 2),
                                  padding='same')
            
            self.net.append(layer)
            if norm_type == 'sbn':
                batch_norm = torch.nn.BatchNorm2d(out_channels)
                self.net.append(batch_norm)
            else:
                raise NotImplementedError
            if i < n_layers - 1:
                dropout = nn.Dropout(1 - keep_prob)
                self.net.append(dropout)
        
    def forward(self, x):
        outs = []
        for layer in self.net:
            x = layer(x)
            outs.append(x)
        return outs

class ClassificationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense3 = nn.Linear(1, 16)
        self.dense4 = nn.Linear(16, 6)
        self.dense2 = nn.Linear(6, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.dense3(x))
        out = self.relu(self.dense4(out))
        out = self.dense2(out)
        return out

                
class FeatureLossBatch(nn.Module):
    def __init__(self, n_layers, base_channels):
        super().__init__()
        self.out_channels = [base_channels * (2 ** (i // 5)) for i in range(n_layers)]
        self.weights = [nn.Parameter(torch.randn(features, 1), requires_grad=True) for features in self.out_channels]

    def forward(self, embeds1, embeds2):
        """
        Both embeds1 and embeeds are outputs from each layer of
        loss_net. 
        a=feat_current[id]-feat_target[id]
        weights = tf.Variable(tf.random_normal([channels[id]]),
                      name="weights_%d" %id, trainable=True)
        a1=tf.transpose(a, [0, 1, 3, 2])
        result=tf.multiply(a1, weights[:,tf.newaxis])
        loss_result=l1_loss_batch(result)
        loss_vec.append(loss_result) 
        """
        loss_vec = []
        for e1, e2 in zip(embeds1, embeds2):
            dist = e1 - e2
            dist = dist.permute(0, 1, 3, 2)
            res = self.weights[i] * dist
            loss = l1_loss_batch_torch(res)
            loss_vec.append(loss)
        return loss_vec, loss_vec


class JNDModel(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.loss_net = LossNet








        




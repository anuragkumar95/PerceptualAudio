import torch
import torch.nn as nn
import torch.nn.functional as F
from network_model_torch import Loss


class JNDTrainer:
    """
    Pytorch recipe to train the JND model described 
    in https://arxiv.org/pdf/2001.04460.pdf
    """
    def __init__(self):
        self.feature_net = 
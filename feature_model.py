import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from model import DGCNN_cls
from PreResNet import ResNet18_1

class FeaModel(nn.Module):
    def __init__(self, output_channels=4):
        super(FeaModel, self).__init__()
        self.dgcnn = DGCNN_cls(output_channels)
        self.resnet = ResNet18_1(output_channels)
        self.fc = nn.Linear(512*4+128, output_channels)
    def forward(self, x, x1, return_fea=False):
        x = self.resnet(x)
        x1 = self.dgcnn(x1)
        x2 = torch.cat((x,x1), dim=1)
        x = self.fc(x2)
        if return_fea:
            return x, x2
        else:
            return x
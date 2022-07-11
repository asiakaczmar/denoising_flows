import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3)
        self.conv2 = torch.nn.Conv2d(16, 16, 3)
        self.conv3 = torch.nn.Conv2d(16, 1, 1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        o = torch.reshape(x, [-1,3,10,10])
        o = self.conv1(o)
        o = self.conv2(o)
        o = self.pool1(o)
        o = self.conv3(o)
        return torch.flatten(o, start_dim=1)

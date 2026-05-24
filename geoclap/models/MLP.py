import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class latlongNet(nn.Module):
    def __init__(self,fc_dim = 64):
        super(latlongNet, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, fc_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        output = self.fc3(x)
        return output
if __name__ == '__main__':
    net = latlongNet(fc_dim=64)
    x = torch.randn(2,3)
    print(net(x).shape) #torch.Size([2, fc_dim])
import torch
import torch.nn as nn

class DnCNN(nn.Module):
    
    def __init__(self, channels=1, num_of_layers=17, bn=True, dropout=False):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        if dropout is True: layers.append(nn.Dropout2d(inplace=True, p=0.1))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            if bn is True: layers.append(nn.BatchNorm2d(features))
            if dropout is True: layers.append(nn.Dropout2d(inplace=True, p=0.5))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        if dropout is True: layers.append(nn.Dropout2d(inplace=True, p=0.5))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, input_x):
        out = self.dncnn(input_x) + input_x
        return out

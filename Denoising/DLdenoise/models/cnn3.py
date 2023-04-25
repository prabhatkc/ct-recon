import torch
import torch.nn as nn
import torch.nn.init as init

class CNN3(nn.Module):
    def __init__(self, num_channels):
        super(CNN3,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=9, padding=4);
        self.relu1 = nn.ReLU();
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1);
        self.relu2 = nn.ReLU();
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, padding=2);
        
        #self._initialize_weights()

    def forward(self,x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        return out

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))

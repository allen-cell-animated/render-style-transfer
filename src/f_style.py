import torch
import torch.nn as nn
import torch.nn.functional as F

from basic_block import BasicBlock
from weight_init import weight_init
### Questions

# 1. This seems very dependent on knowing the starting data size. Is that typical?
# 2. Are there different architectures for different sized images? Ie, more convolutions or more pools or both? How do you decide the shape and size of these layers too.

# Notes:

# 1. More conv layers, and use stride to down sample as alternative to max pool
# 2. adaptive average pooling is better than max pool


# compute a style representation from a collection of images
class FStyle(nn.Module):
    def __init__(self):
        super(FStyle, self).__init__()

        # This results in 2 "parameters": 6 learned kernels of size 5x5 per input channel (3) plus 6 learned scalar bias terms
        self.main = nn.Sequential(
            BasicBlock(3, 8, 5),
            BasicBlock(8, 16, 4, 2),
            BasicBlock(16, 32, 4, 2),
            BasicBlock(32, 64, 4, 2),
            BasicBlock(64, 128, 4, 2),
            BasicBlock(128, 512, 4, 2),
            BasicBlock(512, 512, 4, 2),
            BasicBlock(512, 512, 4, 2)
        )
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

        self.out = nn.Sigmoid()
        self.apply(weight_init)
        

    # running data through the network
    def forward(self, x):
        # batch_size*camera_samples, rgb, x, y

        # input x: torch.Size([64, 3, 392, 392])
        x = self.main(x)

        # torch.Size([64, 16, 19, 19])
        # reshape tensor x to have its second dimension be of size 16*19*19,
        # (to fit into the fc1 ?)
        x = x.view([x.shape[0], -1])
        # torch.Size([64, 5776])
        x = F.relu(self.fc1(x))
        # torch.Size([64, 120])
        x = F.relu(self.fc2(x))
        # torch.Size([64, 84])
        x = self.fc3(x)
        # because fc3 returns length 10, that is our final length of a style representation:
        # torch.Size([64, 10])
        x = self.out(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.conv1 = nn.Conv2d(3, 6, 5)

        # This results in 2x2 downsampling with a "max" filter
        self.pool = nn.MaxPool2d(2, 2)

        # This results in 5x5 downsampling with a "max" filter
        self.pool5 = nn.MaxPool2d(5, 5)

        # This results in 2 "parameters": 16 learned kernels of size 5x5 per input channel (6) plus 16 learned scalar bias terms
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 19 * 19, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # running data through the network
    def forward(self, x):
        # batch_size*camera_samples, rgb, x, y
        # input x: torch.Size([64, 3, 392, 392])
        x = self.pool(F.relu(self.conv1(x)))
        # torch.Size([64, 6, 194, 194])
        x = self.pool(F.relu(self.conv2(x)))
        # torch.Size([64, 16, 95, 95])
        x = self.pool5(x)
        # torch.Size([64, 16, 19, 19])
        # reshape tensor x to have its second dimension be of size 16*19*19,
        # (to fit into the fc1 ?)
        x = x.view(-1, 16 * 19 * 19)
        # torch.Size([64, 5776])
        x = F.relu(self.fc1(x))
        # torch.Size([64, 120])
        x = F.relu(self.fc2(x))
        # torch.Size([64, 84])
        x = self.fc3(x)
        # because fc3 returns length 10, that is our final length of a style representation:
        # torch.Size([64, 10])
        return x

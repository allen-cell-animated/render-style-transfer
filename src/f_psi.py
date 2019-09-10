import torch.nn as nn
import torch.nn.functional as F

from basic_block import BasicBlock
from weight_init import weight_init


# compute a set of render parameters (psi) given an input_data and style
class FPsi(nn.Module):
    def __init__(self, num_render_params=12):
        super(FPsi, self).__init__()
        self.num_render_params = num_render_params
        self.main = nn.ModuleList([
            BasicBlock(3, 8, 5, 1),
            BasicBlock(8, 16, 4, 1),
            BasicBlock(16, 32, 4, 1),
            BasicBlock(32, 64, 4, 1),
            BasicBlock(64, 128, 4, 1),
            BasicBlock(128, 512, 4, 2),
            BasicBlock(512, 512, 4, 2),
            BasicBlock(512, 512, 2, 1)
        ])
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_render_params)

        style_length = 10
        volintensitylength = 1  # n channels in volume ?
        self.styleconv = nn.Linear(style_length, volintensitylength)
        self.apply(weight_init)

    def combine(self, x, y):
        # x is a (batch of) volume data cube, y is a (batch of) style representation of 1xN (1d vector)
        # x is assumed to have one single channel
        # x.shape = [batch, z, y, x]
        # y.shape = [batch, syle_representation_vector_length]

        to_combine = self.styleconv(y)
        # add 2 dimensions and then repeat values.
        expanded = to_combine.unsqueeze(2).unsqueeze(2).repeat([1, x.shape[1], x.shape[2], x.shape[3]])
        return x + expanded

    # running data through the network
    def forward(self, x, y):
        # getting shapes right for inputs into f_psi:
        # psis shape: torch.Size([4, 3, 3])
        # flattened batch of psis torch.Size([360])
        # Y: batch of styles torch.Size([4, 10])
        # X: im_cube shape torch.Size([4, 3, 392, 392])

        # f_psi will take a batch of num_camera_samples styles and a batch of 1 data cube
        # and return a batch of num_camera_samples psis

        # a batch (4) of 3x3 filters
        # want to return shape[4, 3, 3]

        # x is the volume data
        # y is the style tensor

        # strategy :  combine the style with the volume, then downsample, then repeat a few times.

        for layer in self.main:
            x = self.combine(x, y)
            x = layer(x)

        # reshape tensor x to have its second dimension be of size 16*19*19,
        # (to fit into the fc1 ?)
        x = x.view([x.shape[0], -1])

        # torch.Size([4, 5776])
        x = F.relu(self.fc1(x))
        # torch.Size([4, 120])
        x = F.relu(self.fc2(x))
        # torch.Size([4, 84])
        x = self.fc3(x)
        # because fc3 returns length num_render_params, that is our final length of a style representation:
        # torch.Size([4, num_render_params])
        return x

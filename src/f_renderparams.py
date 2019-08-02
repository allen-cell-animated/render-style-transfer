import torch
import torch.nn as nn
import torch.nn.functional as F

### F\_Psi questions

# 1. For FPsi, if we do a convolution network, how do we incorporate the target style tensor into the input data?
# 2. Could we do something where we pick random parameters, render an image, get the style, compare with target style and then adjust?

# compute a set of render parameters (psi) given an input_data and style
class FPsi(nn.Module):
    def __init__(self, num_render_params=9):
        # TODO IMPLEMENT greg hints:
        # 4 downsampling conv2ds, and then adaptive average pooling to get to the final result size
        super(FPsi, self).__init__()
        # args: in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'
        # This results in 2 "parameters": 6 learned kernels of size 5x5 per input channel (3) plus 6 learned scalar bias terms
        self.conv1 = nn.Conv2d(3, 6, 5)

        # Applies a 2D max pooling over an input signal composed of several input planes.
        # args: (kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        # This results in 2x2 downsampling with a "max" filter
        self.pool = nn.MaxPool2d(2, 2)
        self.pool19 = nn.MaxPool2d(19, 19)

        # This results in 2 "parameters": 16 learned kernels of size 5x5 per input channel (6) plus 16 learned scalar bias terms
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Applies a linear transformation to the incoming data: y = xA^T + b
        # args: (in_features, out_features, bias=True)
        # This results in 2 "parameters": 120 learned linear weight vectors of length 16*5*5, plus 120 learned scalar bias terms
        self.fc1 = nn.Linear(16 * 19 * 19, 120)
        # This results in 2 "parameters": 84 learned linear weight vectors of length 120, plus 84 learned scalar bias terms
        self.fc2 = nn.Linear(120, 84)
        # This results in 2 "parameters": 10 learned linear weight vectors of length 84, plus 10 learned scalar bias terms
        self.fc3 = nn.Linear(84, num_render_params)
        # self.combo = CombiningNetwork(x, y)

    def combine(self, x, y):
        import pdb

        pdb.set_trace()
        # x: a = torch.Tensor([[0,1,2],[3,4,5],[6,7,8]])
        x1 = x.clone()
        x1.unsqueeze_(-1)
        sizetuple = list(x1.shape)
        # change last element of sizetuple to be 10
        # expect
        size_of_style_tensor = y.shape
        ylength = size_of_style_tensor[0]
        sizetuple[len(sizetuple) - 1] = ylength
        z = x1.expand(sizetuple)
        w = z + y
        return w

    # running data through the network
    def forward(self, x, y):
        # getting shapes right for inputs into f_psi:
        # psis shape: torch.Size([4, 16, 3, 3])
        # flattened batch of psis torch.Size([360])
        # Y: batch of styles torch.Size([64, 10])
        # X: im_cube shape torch.Size([4, 3, 392, 392])

        # f_psi will take a batch of 16 styles and a batch of 1 data cube
        # and return a batch of 16 psis

        # a batch (4) of 16 3x3 filters
        # want to return shape[64, 3, 3] or [4, 16, 3, 3] ?

        # x is the volume data
        # y is the style tensor

        x = self.combine(x, y)
        # [ 4, 3, 392, 392, 64, 10]

        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool19(F.relu(self.conv2(x)))
        # reshape tensor x to have its second dimension be of size 16*5*5,
        # (to fit into the fc1 ?)
        x = x.view(-1, 10)

        y = self.y_subnet(y)
        ## GREG hints: just add stuff together.
        x = F.relu(self.fc1(x + y))
        x = F.relu(self.fc2(x))

        x = self.fc3(x + y)
        return x
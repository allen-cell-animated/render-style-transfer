import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


# returns a pil-compatible rbg image
def render_function(input_data, render_params, camera_transform):
    # this function could do anything (e.g. do volume rendering)
    # in this case, the "render" will be a convolution filter and the render params are the filter weights
    # and our camera_transform just rotates the image

    # the first 9 render_params need to become a 3x3 filter to apply to the image.
    if len(render_params) < 9:
        raise "Bad number of render_params"
    convfilter = [[render_params[i + j * 3] for i in range(3)] for j in range(3)]

    # note that groups=3 and we only put one in the array here
    # this is reshaping to make the conv2d happy
    reshaped_params = [convfilter]
    renderedimage = F.conv2d(
        torch.Tensor([input_data]),
        # apply same filter to each of r,g,b channels
        torch.Tensor([reshaped_params, reshaped_params, reshaped_params]),
        bias=None,
        stride=1,
        padding=1,
        dilation=1,
        groups=3,
    )
    # normalize back to displayable range after convolution
    maxval = renderedimage.max()
    renderedimage = transforms.functional.normalize(
        renderedimage[0], mean=[0, 0, 0], std=[maxval, maxval, maxval]
    )
    renderedimage_pil = transforms.functional.to_pil_image(renderedimage, mode=None)

    final_image = camera_transform(renderedimage_pil)

    return final_image

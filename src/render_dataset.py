import os
import random

from skimage import io
import PIL
# from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Setting up dataset and data loader

# Our dataset transforms the data on the fly.

# It has a render method that takes in input image data, render params and camera restrictions.

# It chooses camera_samples random "camera" rotations, and creates camera_samples number of images of the same "data cube" with the one set of render parameters.

# get item returns:

# 1. im_as_tensor (input data cube as tensor)
# 2. images (camera_samples images)
# 3. im_2d_cube_ids (array of length camera_samples, all with the same cube id)
# 4. render_params (array of length camera_samples, all with the same render parameters)


class RenderStyleTransferDataset(Dataset):
    def __init__(self, root_dir, camera_samples=16, train=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            camera_samples(number): how many images to return per __get_item__
            train (boolean): TODO: select images from training set vs test set
        """

        # TODO point this to /allen file system data sets
        self.root_dir = root_dir

        self.camera_samples = camera_samples
        self.all_files = []
        for name in os.listdir(self.root_dir):
            full = os.path.join(self.root_dir, name)
            if os.path.isfile(full) and name[-4:] == ".png":
                self.all_files.append(name)
        num_files = len(self.all_files)
        print("RenderStyleTransferDataset created with size: " + str(num_files))

    def __len__(self):
        return len(self.all_files)

    # returns a Tensor that is a 2D RGB image
    def render_function(self, input_data, render_params, camera_transform):
        # this function could do anything (e.g. do volume rendering)
        # in this case, the "render" will be a convolution filter and the render params are the filter weights
        # and our camera_transform just rotates the image

        # note that groups=3 and we only put one in the array here
        # this is reshaping to make the conv2d happy
        reshaped_params = [render_params]
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

        final_image = transforms.functional.to_tensor(final_image)
        return final_image

    def __getitem__(self, idx):
        # load some input_data for our render_function
        img_name = os.path.join(self.root_dir, self.all_files[idx])
        image = io.imread(img_name)

        # generate a repeatable set of render parameters for our render_function
        random.seed(a=idx)
        convfilter = [[random.random() for i in range(3)] for j in range(3)]

        # prepare the sampler of camera transforms
        camera_degree_range = 45.0
        apply_camera = transforms.RandomRotation(
            camera_degree_range, resample=PIL.Image.BICUBIC
        )

        # loop to generate a set of images with the same style (render settings) but different camera angles
        images = []
        for i in range(self.camera_samples):
            # generate a rendered image of the given style
            renderedimage = self.render_function(
                image.transpose(2, 1, 0), convfilter, apply_camera
            )
            images.append(renderedimage)
        images = torch.stack(images)
        im_2d_cube_ids = torch.Tensor([idx for i in range(self.camera_samples)])
        render_params = torch.stack(
            [torch.Tensor(convfilter) for i in range(self.camera_samples)]
        )

        im_as_tensor = transforms.functional.to_tensor(image)

        # one item from this data set consists of:
        #   an input data image,
        #   a set of images generated with the same render_parameters("style") but different camera angles,
        #   a set of indices that will tie the rendered images back to the data image,
        #   the set of render_parameters
        sample = (im_as_tensor, images, im_2d_cube_ids, render_params)

        return sample


# testDataset = RenderStyleTransferDataset(root_dir="/thumbnail-dataset")
# item = testDataset.__getitem__(0)
# print(item[0].shape)
# print(item[1].shape)
# print(item[2].shape)
# print(item[3].shape)

# imshow_list([item[1][i] for i in range(10)])

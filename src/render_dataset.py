import itertools
import os
import random

from skimage import io
import PIL

# from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from render_function import render_function

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

        training_set_count = int(len(self.all_files) * 0.8)
        test_set_count = len(self.all_files) - training_set_count
        if train:
            self.all_files = self.all_files[:training_set_count]
        else:
            self.all_files = self.all_files[-test_set_count:]

        num_files = len(self.all_files)
        print("RenderStyleTransferDataset created with size: " + str(num_files))

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        # load some input_data for our render_function
        img_name = os.path.join(self.root_dir, self.all_files[idx])
        image = io.imread(img_name)

        # generate a repeatable set of render parameters for our render_function
        random.seed(a=idx)
        convfilter = [random.random() for i in range(9)]

        # prepare the sampler of camera transforms
        camera_degree_range = 45.0
        apply_camera = transforms.RandomRotation(
            camera_degree_range, resample=PIL.Image.BICUBIC
        )

        # loop to generate a set of images with the same style (render settings) but different camera angles
        images = []
        for i in range(self.camera_samples):
            # generate a rendered image of the given style
            renderedimage = render_function(
                image.transpose(2, 1, 0), convfilter, apply_camera
            )
            final_image = transforms.functional.to_tensor(renderedimage)
            images.append(final_image)
        images = torch.stack(images)
        im_2d_cube_ids = torch.Tensor([idx for i in range(self.camera_samples)])
        render_params = torch.Tensor(convfilter)

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

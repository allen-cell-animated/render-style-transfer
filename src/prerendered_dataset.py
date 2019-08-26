import json
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
    def __init__(self, cache_file, train=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            camera_samples(number): how many images to return per __get_item__
            train (boolean): TODO: select images from training set vs test set
        """

        self.cache_file = cache_file
        with open(cache_file, "r") as read_file:
            self.dataset = json.load(read_file)

        count = len(self.dataset)
        training_set_count = int(count * 0.8)
        test_set_count = count - training_set_count
        if train:
            self.dataset = self.dataset[:training_set_count]
        else:
            self.dataset = self.dataset[-test_set_count:]

        self.num_files = len(self.dataset)
        print("RenderStyleTransferDataset created with size: " + str(self.num_files))

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        # load some input_data for our render_function
        dataset_entry = self.dataset[idx]
        img_name = os.path.join(self.root_dir, self.all_files[idx])
        image = io.imread(dataset_entry["data_file"])

        convfilter = dataset_entry["render_params"]

        # loop to generate a set of images with the same style (render settings) but different camera angles
        images = []
        for i, path in enumerate(dataset_entry["renders"]):
            # generate a rendered image of the given style
            renderedimage = io.imread(path)
            final_image = transforms.functional.to_tensor(renderedimage)
            images.append(renderedimage)
        images = torch.stack(images)
        im_2d_cube_ids = torch.Tensor(
            [idx for i in range(len(dataset_entry["renders"]))]
        )
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

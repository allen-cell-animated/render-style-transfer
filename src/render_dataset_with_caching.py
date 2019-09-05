import itertools
import os
import random
import json

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

# cache_setting options: load (load from cache), save (save results of rendering), none (do nothing)


class StyleTransferDataset(Dataset):
    def __init__(self, data_dir, camera_samples=16, num_psis_per_data_cube=2, cache_setting="save", cache_dir="cached", train=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            camera_samples(number): how many images to return per __get_item__
            train (boolean): TODO: select images from training set vs test set
        """

        # TODO point this to /allen file system data sets
        self.data_dir = data_dir
        self.cache_file = f"{cache_dir}/dataset.json"
        self.cache_dir = cache_dir
        self.camera_samples = camera_samples
        self.cache_setting = cache_setting
        self.num_psis_per_data_cube = num_psis_per_data_cube

        if self.cache_setting == "load":
            print('loading file list from cache')
            self.load_from_dataset_file()

        else:
            print('loading file list from directory')
            self.load_from_data_dir()
            if self.cache_setting == "save":
                with open(f"{cache_dir}/dataset.json", "w") as fout:
                    json.dump([], fout)

        count = len(self.all_files)
        training_set_count=int(count * 0.8)
        test_set_count = len(self.all_files) - training_set_count
        
        if train:
            self.all_files = self.all_files[:training_set_count]
        else:
            self.all_files = self.all_files[-test_set_count:]

        self.num_files = len(self.all_files)
        print("RenderStyleTransferDataset created with size: " + str(self.num_files))

    def load_from_dataset_file(self): 
        with open(self.cache_file, "r") as read_file:
            self.dataset = json.load(read_file)
            self.all_files = list(map(lambda x: x["data_file"], self.dataset))

    def load_from_data_dir(self):
        self.all_files = []
        for name in os.listdir(self.data_dir):
            full = os.path.join(self.data_dir, name)
            if os.path.isfile(full) and name[-4:] == ".png":
                self.all_files.append(name)

    def save_to_dataset_file(self, img_name, images_names, render_params, render_ids):
        with open(self.cache_file, "r") as read_file:
            dataset = json.load(read_file)

        dataset.append({"data_file": img_name, "renders": images_names,
                        "render_params": render_params, "render_ids": render_ids})

        with open(f"{self.cache_dir}/dataset.json", "w") as fout:
            json.dump(dataset, fout)

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        # load some input_data for our render_function
        im_2d_cube_ids = []
        images = []

        if self.cache_setting == "load":
            dataset_entry = self.dataset[idx]
            image = io.imread(dataset_entry["data_file"])
            render_params = dataset_entry["render_params"]
            render_ids = dataset_entry["render_ids"]
            renders = dataset_entry["renders"]

            # loop to generate a set of images of length: len(render_ids) *  camera_samples
            for i, path in enumerate(renders):
                # generate a rendered image of the given style
                renderedimage = io.imread(path)
                final_image = transforms.functional.to_tensor(renderedimage)
                images.append(final_image)
                im_2d_cube_ids.append(
                    idx + render_ids[i//int(self.camera_samples)])

        else: 
            img_name = os.path.join(self.data_dir, self.all_files[idx])
            image = io.imread(img_name)
            render_params = []
            render_ids = [i for i in range(self.num_psis_per_data_cube)]
            images_names = []

            # prepare the sampler of camera transforms
            camera_degree_range = 45.0
            apply_camera = transforms.RandomRotation(
                camera_degree_range, resample=PIL.Image.BICUBIC
            )

            # loop to generate a set of images with different styles and different camera angles
            for k in range(self.num_psis_per_data_cube):
                # generate a repeatable set of render parameters for our render_function
                random.seed(a=(idx + k))
                convfilter = [random.random() for i in range(9)]

                for j in range(self.camera_samples):
                    # generate a rendered image of the given style
                    renderedimage = render_function(image.transpose(2, 1, 0), convfilter, apply_camera)
                    final_image = transforms.functional.to_tensor(renderedimage)
                    images.append(final_image)
                    im_2d_cube_ids.append(idx + render_ids[j//int(self.camera_samples)])
                    render_params.append(convfilter)
                    
                    if self.cache_setting == "save":
                        outpath = f"{self.cache_dir}/{self.all_files[idx]}_rendered_{j}_{k}.png"
                        images_names.append(outpath)
                        renderedimage.save(outpath)


            if self.cache_setting == "save":
                self.save_to_dataset_file(img_name, images_names, render_params, render_ids)
    
        images = torch.stack(images)
        im_as_tensor = transforms.functional.to_tensor(image)
        im_2d_cube_ids = torch.Tensor(im_2d_cube_ids)
        render_params = torch.Tensor(render_params)

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

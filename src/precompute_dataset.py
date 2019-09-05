# Precompute data set from source data, and save the rendered images and data set to file
import argparse
import json
import os
from skimage import io
import PIL
import random

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from render_dataset_with_caching import StyleTransferDataset
from render_function import render_function

def img_from_tensor(t):
    return transforms.functional.to_pil_image(t, mode=None)


def main(
    data_dir="D:/src/aics/render-style-transfer/training_data",
    camera_samples=16,
    num_psis_per_data_cube=2,
    cache_dir="cached",
):
    train_dataset = StyleTransferDataset(
        data_dir, camera_samples, num_psis_per_data_cube, "save", cache_dir, True)

    for i, _ in enumerate(train_dataset.all_files):
        train_dataset.__getitem__(i)
    
    test_dataset = StyleTransferDataset(
            data_dir, camera_samples, num_psis_per_data_cube, "save", cache_dir, False)

    for j, _ in enumerate(test_dataset.all_files):
        test_dataset.__getitem__(j)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cache dataset options')
    parser.add_argument(
        '--data_dir',
        default='D:/src/aics/render-style-transfer/training_data',
        help='provide provide the training data directory'
    )
    parser.add_argument(
        '--cache_dir',
        default='cached',
        help='provide an directory for the cache, default cached'
    )
    parser.add_argument(
        '--num_styles_per_image',
        default=1,
        help='number of psis per data cube'
    )
    parser.add_argument(
        '--num_camera_renders',
        default=16,
        help='number of camera rotations'
    )

    options = parser.parse_args()
    print('dataset options', options)
    main(options.data_dir, options.num_camera_renders, options.num_styles_per_image,
         options.cache_dir)

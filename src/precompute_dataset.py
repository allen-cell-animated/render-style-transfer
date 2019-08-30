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

from render_function import render_function

# Utility functions


def img_from_tensor(t):
    return transforms.functional.to_pil_image(t, mode=None)


def main(
    num_renders=16,
    src_dir="D:/src/aics/render-style-transfer/training_data",
    cache_dir="cached",
):
    dataset = []
    # collect up all file paths of png files in src_dir
    all_files = []
    all_files_names = []
    for name in os.listdir(src_dir):
        full = os.path.join(src_dir, name)
        if os.path.isfile(full) and name[-4:] == ".png":
            all_files.append(full)
            all_files_names.append(name[:-4])

    os.makedirs(cache_dir, exist_ok=True)

    num_files = len(all_files)
    print(f"found {num_files} data files")

    for i, data_file in enumerate(all_files):
        filename = all_files_names[i]
        print(data_file)
        image = io.imread(data_file)

        # generate a repeatable set of render parameters for our render_function
        random.seed(a=i)
        convfilter = [random.random() for k in range(9)]

        # prepare the sampler of camera transforms
        camera_degree_range = 45.0
        apply_camera = transforms.RandomRotation(
            camera_degree_range, resample=PIL.Image.BICUBIC
        )

        # loop to generate a set of images with the same style (render settings) but different camera angles
        images = []
        for j in range(num_renders):
            # generate a rendered image of the given style
            renderedimage = render_function(
                image.transpose(2, 1, 0), convfilter, apply_camera
            )
            outpath = f"{cache_dir}/{filename}_rendered_{j}.png"
            renderedimage.save(outpath)
            images.append(outpath)

        # one item from this data set consists of:
        #   an input data image,
        #   a set of images generated with the same render_parameters("style") but different camera angles,
        #   the set of render_parameters
        dataset.append(
            {"data_file": data_file, "renders": images, "render_params": convfilter}
        )

    with open(f"{cache_dir}/dataset.json", "w") as fout:
        json.dump(dataset, fout)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cache dataset options')
    parser.add_argument(
        '--src_dir',
        default='D:/src/aics/render-style-transfer/training_data',
        help='provide provide the training data directory'
    )
    parser.add_argument(
        '--cache_dir',
        default='cached',
        help='provide an directory for the cache, default cached'
    )
    options = parser.parse_args()
    print('in out folders', options)
    main(16, options.src_dir, options.cache_dir)

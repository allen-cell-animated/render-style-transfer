# Render style transfer
import argparse
import numpy
import os
import torch
import torchvision
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt

from dataset import StyleTransferDataset
from render_function import render_function

from f_style import FStyle
from f_psi import FPsi

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"


# pil image or ndarray
def imshow(img):
    plt.figure()
    plt.subplots()
    plt.axis("off")
    plt.imshow(img)
    plt.show()


# list of tensors
def imshow_list(img):
    st = torch.stack(img)
    imshow(torchvision.utils.make_grid(st).permute(1, 2, 0))


def img_from_tensor(t):
    return transforms.functional.to_pil_image(t, mode=None)


def randomly_choose(list_of_stuff):
    perm = torch.randint(list_of_stuff.size(0), (1,))
    return list_of_stuff[perm]


def test(version, testloader):
    f_style, f_psi = load_model(version)

    print("test")
    num_camera_samples = testloader.dataset.camera_samples

    # test trained model:

    dataiter = iter(testloader)
    # grab batch of four images again but from data it hasn't seen before
    im_cube, im_2d, im_style_ids, datacube_ids, psi = dataiter.next()
    print(f"im_cube shape: {im_cube.shape}")
    print(f"im_2d shape: {im_2d.shape}")
    print(f"im_style_ids shape: {im_style_ids.shape}")
    print(f"datacube_ids shape: {datacube_ids.shape}")
    print(f"psi shape: {psi.shape}")

    #
    im_cube = im_cube.to(device)
    im_2d = im_2d.to(device)
    im_style_ids = im_style_ids.to(device)
    datacube_ids = datacube_ids.to(device)
    psi = psi.to(device)

    # loop to generate a set of images with different styles and different camera angles
    the_render_function = render_function()

    num_camera_samples = testloader.dataset.camera_samples
    num_psis_per_data_cube = testloader.dataset.num_psis_per_data_cube

    with torch.no_grad():
        flattened_im = torch.flatten(im_2d, 0, 1)
        batch_of_styles = f_style(flattened_im)

        # choose a batch of styles to pass to f_psi
        # by picking one from each set of renders that came from one data cube

        # pick a random camera angle from each data cube:
        # each data cube has (num_camera_samples*num_psis_per_data_cube) rendered images in im_2d
        # pick a random int up to (num_camera_samples*num_psis_per_data_cube), and do it batch_size times
        perm = torch.randint(num_camera_samples * num_psis_per_data_cube, (testloader.batch_size,))
        # loop over all data cubes?
        for j in range(testloader.batch_size):
            perm[j] += j * num_camera_samples * num_psis_per_data_cube

        # a group of styles for each data cube in the batch???
        small_batch_of_styles = batch_of_styles[perm]
        small_batch_of_images = flattened_im[perm]
        computed_psi = f_psi(im_cube, small_batch_of_styles)
        print(f"computed psi shape : {computed_psi.shape}")

        subbatchpsi = psi[perm]
        for i in range(computed_psi.shape[0]):
            # TODO fix indexing of psi
            t = torch.dist(computed_psi[i], psi[i])
            print(f"Distance between computed psi and actual psi: {t}")
            print(f"Learned psi: {computed_psi[i]}")

        styleA = small_batch_of_styles[0]
        # psiA = computed_psi[0]
        distancelist = [torch.dist(styleA, small_batch_of_styles[j]).item() for j in range(len(small_batch_of_styles))]
        sorted_indices = numpy.argsort(distancelist)

        print(f"flattened_im shape: {flattened_im.shape}")
        print(f"batch_of_styles shape: {batch_of_styles.shape}")
        print(f"small_batch_of_styles shape: {small_batch_of_styles.shape}")
        print(f"im_cube shape: {im_cube.shape}")
        print(f"computed_psi shape: {computed_psi.shape}")

        print("SORTED STYLE DIFFERENCES AGAINST SYLE OF IMAGE 0:")
        print(sorted(distancelist))
        for i in range(small_batch_of_styles.shape[0]):
            print(f"Learned style: {small_batch_of_styles[i]}")

        for m, k in enumerate(sorted_indices):
            params = the_render_function.denormalize_render_params(computed_psi[k])
            # generate a rendered image of the given style
            renderedimage = the_render_function.render(
                transforms.functional.to_pil_image(im_cube[k]), params, None)
            outpath = f"results/TESTRESULT_{m}.png"
            renderedimage.save(outpath)
            outpath = f"results/GROUNDTRUTH_{m}.png"
            transforms.functional.to_pil_image(small_batch_of_images[k]).save(outpath)


def load_model(version):
    # load an existing model
    f_style = FStyle()
    f_style.load_state_dict(torch.load(f"./model/fstyle_{version}"))
    f_style.eval()
    f_style.to(device)

    f_psi = FPsi()
    f_psi.load_state_dict(torch.load(f"./model/fpsi_{version}"))
    f_psi.eval()
    f_psi.to(device)

    return (f_style, f_psi)


def main(loss_file_name, keep_logs, use_cached=True, model_version=None):
    # mac_data_dir = "/Volumes/aics/animated-cell/Dan/renderstyletransfer/training_data"
    # pc_data_dir = "//allen/aics/animated-cell/Dan/renderstyletransfer/training_data"
    # windows_local_data_dir = "D:/src/aics/render-style-transfer/training_data"

    data_dir = "D:/src/aics/render-style-transfer/training_data"
    batch_size = 8

    testset = StyleTransferDataset(
        data_dir, cache_setting="load" if use_cached else "none", train=False)

    # takes the dataset we defined, loads a batch at a time,
    # shuffle=True reshuffles the data every epoch
    # for shuffle, an epoch is defined as one full iteration through the DataLoader
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    # get latest modified model version if none is specified
    if model_version is None:
        def all_files_under(path):
            """Iterates through all files that are under the given path."""
            for cur_path, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    yield "model/" + filename

        latest_file = max(all_files_under('model'), key=os.path.getmtime)
        model_version = latest_file.split("_")[1]

    test(model_version, testloader)

    print("Finished Testing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Style transfer options')
    parser.add_argument(
        '--save_loss_file',
        default=False,
        help='if set to true, will check in result to git'
    )
    parser.add_argument(
        '--loss_file_name',
        default=time.time_ns(),
        help='file name of loss log'
    )
    parser.add_argument(
        '--ignore_cache',
        action='store_true',
        help='load dataset from precached data'
    )
    parser.add_argument(
        '--model',
        default=None,
        help='model name to load for testing'
    )
    options = parser.parse_args()
    print('loss log options', options)
    print("cuda:0" if torch.cuda.is_available() else "cpu")
    main(options.loss_file_name, options.save_loss_file, not options.ignore_cache, options.model)

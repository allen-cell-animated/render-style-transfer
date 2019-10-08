# Render style transfer
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import pathlib

from dataset import StyleTransferDataset

from f_style import FStyle
from f_psi import FPsi

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def train(
    f_style,
    f_psi,
    trainloader,
    loss_file_name,
    keep_logs=False,
    version="",
    learning_rate=0.0001,
    number_of_epochs=1
):
    num_camera_samples = trainloader.dataset.camera_samples
    num_psis_per_data_cube = trainloader.dataset.num_psis_per_data_cube

    print("camera samples: " + str(num_camera_samples))
    print("number of render params per data cube: " + str(num_psis_per_data_cube))

    if keep_logs:
        path = 'results'
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    else:
        path = 'loss'
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    file_path = f"{path}/{loss_file_name}.csv"
    f = open(file_path, "w+")
    f.write("Epoch, index,loss_style,loss_psi,total_loss\n")
    f.close()
    loss_fn = nn.MSELoss()
    regularization_rate = 1000  # aka lambda
    # Implements stochastic gradient descent (optionally with momentum).
    optimizer = optim.Adam(f_style.parameters(), lr=learning_rate)
    for epoch in range(number_of_epochs):  # loop over the dataset multiple times
        print(f"epoch {epoch}")
        running_loss = 0.0
        # trainloader is the method that gets 4 examples out of the training dataset at a time
        # each epoch trainloader will shuffle the data because we told it to
        for i, data in enumerate(trainloader, 0):
            im_cube, im_2d, im_style_ids, datacube_ids, psi = data

            # batch shape debugging
            # print(f"im_cube shape: {im_cube.shape}")
            # print(f"im_2d shape: {im_2d.shape}")
            # print(f"im_style_ids shape: {im_style_ids.shape}")
            # print(f"psi shape: {psi.shape}")

            im_cube = im_cube.to(device)
            im_2d = im_2d.to(device)
            im_style_ids = im_style_ids.to(device)
            datacube_ids = datacube_ids.to(device)
            psi = psi.to(device)

            # batch_size * num_camera_samples * num_psis_per_data_cube, scalar
            batch_of_style_ids = torch.flatten(im_style_ids)
            batch_of_datacube_ids = torch.flatten(datacube_ids)

            # batch_size * num_camera_samples * num_psis_per_data_cube, num of render params
            batch_of_psis = torch.flatten(psi, 0, 1)

            # batch_size * num_camera_samples * num_psis_per_data_cube, dimensions of rendered image
            flattened_im = torch.flatten(im_2d, 0, 1)

            batch_of_styles = f_style(flattened_im)

            ############################################
            # choose a batch of styles to pass to f_psi
            # by picking one from each sub-batch

            # pick a random camera angle from each data cube:
            # each data cube has (num_camera_samples*num_psis_per_data_cube) rendered images in im_2d
            # pick a random int up to (num_camera_samples*num_psis_per_data_cube), and do it batch_size times
            perm = torch.randint(num_camera_samples * num_psis_per_data_cube, (trainloader.batch_size,))
            # adjust the offsets by how far we are in the batch of data cubes
            # there are (num_camera_samples * num_psis_per_data_cube) entries per data cube
            for j in range(trainloader.batch_size):
                perm[j] += j * num_camera_samples * num_psis_per_data_cube

            # one style per data cube in the batch
            small_batch_of_styles = batch_of_styles[perm]

            # print(f"batch_of_style_ids shape: {batch_of_style_ids.shape}")
            # print(f"batch_of_datacube_ids shape: {batch_of_datacube_ids.shape}")
            # print(f"batch_of_psis shape: {batch_of_psis.shape}")
            # print(f"flattened_im shape: {flattened_im.shape}")
            # print(f"batch_of_styles shape: {batch_of_styles.shape}")
            # print(f"small_batch_of_styles shape: {small_batch_of_styles.shape}")
            # print(f"im_cube shape: {im_cube.shape}")

            psi_hat = f_psi(im_cube, small_batch_of_styles)

            import pdb; pdb.set_trace()
            loss_psi = loss_fn(psi_hat, batch_of_psis[perm])
            ###########################################

            loss_style = torch.zeros(1).to(device)

            # same style means same data cube and same render params but different camera transform
            for j, s in enumerate(batch_of_styles):
                current_style_id = batch_of_style_ids[j]
                current_datacube_id = batch_of_datacube_ids[j]
                # print('current_style_id', current_style_id)

                # get all indices of ids that are the same as j
                list_of_same_ids = (batch_of_style_ids == current_style_id).nonzero().flatten()
                # print('list_of_same_ids', list_of_same_ids)

                # pick one
                # (sometimes this will be j, and the distance will be 0.  is that ok?)
                index_with_id_same = randomly_choose(list_of_same_ids)

                # get all indices of ids that are different than j
                different_styles = (batch_of_style_ids != current_style_id)

                same_datacubes = (batch_of_datacube_ids == current_datacube_id)
                diff_datacubes = (batch_of_datacube_ids != current_datacube_id)
                
                intersection1 = different_styles * same_datacubes
                intersection2 = diff_datacubes

                list1 = intersection1.nonzero().flatten()
                idx1 = randomly_choose(list1)
                list2 = intersection2.nonzero().flatten()
                idx2 = randomly_choose(list2)

                #print(idx1)
                #print(idx2)

                list_of_different_style_ids = torch.LongTensor([idx1, idx2])
                ####
                # TODO: ?filter this so that we only pick ids from same data cube as j came from?
                # (this should help it learn that style is not the same as data cube)
                ####
                # print('list of different', list_of_different_style_ids)

                # pick one
                index_with_id_different = randomly_choose(list_of_different_style_ids)
                #print(f'different choice : {index_with_id_different}')
                #print(f'batch_of_styles : {batch_of_styles.shape}')
                # compute loss
                loss_style = (
                    loss_style + torch.dist(s, batch_of_styles[index_with_id_same]) ** 2 - torch.dist(s, batch_of_styles[index_with_id_different]) ** 2
                )

            # print("new total loss style", loss_style)
            loss_style = loss_style / len(batch_of_styles)
            # print("new average loss style", loss_style)
            total_loss = regularization_rate * loss_psi + loss_style
            # print("new average loss psi", loss_psi)
            # print(f"loss: {loss_style.item()} + {regularization_rate*loss_psi.item()} : total {total_loss.item()}")

            # append current loss log with results
            f = open(file_path, "a+")
            f.write(f"{epoch},{i},{loss_style.item()},{regularization_rate*loss_psi.item()},{total_loss.item()}\n")
            f.close()

            total_loss.backward()

            optimizer.step()

            # print statistics
            running_loss += total_loss.item()
            everyN = 10
            if i % everyN == everyN - 1:  # print every 10 mini-batches
                print(f"[{epoch+1}, {i+1}] loss: {running_loss/everyN}")
                running_loss = 0.0
    if version == "":
        version = str(time.time_ns())
    save_model(f_style, f_psi, version)


def save_model(f_style, f_psi, version):
    # save a trained model
    pathlib.Path("./model/").mkdir(parents=True, exist_ok=True)
    torch.save(f_style.state_dict(), f"./model/fstyle_{version}.pt")
    torch.save(f_psi.state_dict(), f"./model/fpsi_{version}.pt")


def main(loss_file_name, keep_logs, use_cached=True):
    # mac_data_dir = "/Volumes/aics/animated-cell/Dan/renderstyletransfer/training_data"
    # pc_data_dir = "//allen/aics/animated-cell/Dan/renderstyletransfer/training_data"
    # windows_local_data_dir = "D:/src/aics/render-style-transfer/training_data"

    data_dir = "D:/src/aics/render-style-transfer/training_data"
    batch_size = 4
    learning_rate = 0.0001
    number_of_epochs = 1
    model_version_name = "model0"

    trainset = StyleTransferDataset(data_dir, cache_setting="load" if use_cached else "none", train=True)

    # takes the trainset we defined, loads 4 (default 1) at a time,
    # shuffle=True reshuffles the data every epoch
    # for shuffle, an epoch is defined as one full iteration through the DataLoader
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    # Training

    f_style = FStyle().to(device)
    f_psi = FPsi().to(device)

    train(f_style, f_psi, trainloader, loss_file_name, keep_logs, model_version_name, learning_rate, number_of_epochs)

    print("Finished Training")


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
    options = parser.parse_args()
    print('loss log options', options)
    print("cuda:0" if torch.cuda.is_available() else "cpu")
    main(options.loss_file_name, options.save_loss_file, not options.ignore_cache)

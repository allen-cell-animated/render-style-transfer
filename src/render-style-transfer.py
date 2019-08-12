# Render style transfer

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from render_dataset import RenderStyleTransferDataset
from f_style import FStyle
from f_renderparams import FPsi

# Utility functions

# functions to show an image


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


def main():
    # Data loader. Combines a dataset and a sampler, and provides single- or multi-process iterators over the dataset.

    # train (bool, optional)
    # split the data set based on the value of train into a repeatable 80%-20% split
    #trainset = RenderStyleTransferDataset(root_dir="//allen/aics/animated-cell/Dan/renderstyletransfer/training_data", train=True)
    #testset = RenderStyleTransferDataset(root_dir="//allen/aics/animated-cell/Dan/renderstyletransfer/training_data", train=False)
    trainset = RenderStyleTransferDataset(root_dir="D:/src/aics/render-style-transfer/training_data", train=True)
    testset = RenderStyleTransferDataset(root_dir="D:/src/aics/render-style-transfer/training_data", train=False)

    # takes the trainset we defined, loads 4 (default 1) at a time,
    # shuffle=True reshuffles the data every epoch
    # for shuffle, an epoch is defined as one full iteration through the DataLoader
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=0
    )

    # same as trainloader
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=0
    )

    # Training

    f_style = FStyle().cuda()
    f_psi = FPsi().cuda()

    loss_fn = nn.MSELoss()
    regularization_rate = 0.1  # aka lambda
    # Implements stochastic gradient descent (optionally with momentum).
    # lr is the learning rate, required
    optimizer = optim.Adam(f_style.parameters(), lr=0.001)
    number_of_epochs = 1

    for epoch in range(number_of_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        # trainloader is the method that gets 4 examples out of the training dataset at a time
        # each epoch trainloader will shuffle the data because we told it to
        for i, data in enumerate(trainloader, 0):
            im_cube, im_2d, im_2d_cube_id, psi = data

            im_cube = im_cube.cuda()
            im_2d = im_2d.cuda()
            im_2d_cube_id = im_2d_cube_id.cuda()
            psi = psi.cuda()

            batch_of_ids = torch.flatten(im_2d_cube_id)
            # print('batch_of_psis shape:', psi.shape)

            # combine so that the batch is really batch_size*num_camera_samples
            flattened_im = torch.flatten(im_2d, 0, 1)
            batch_of_styles = f_style(flattened_im)
            # print('batch of styles', batch_of_styles.shape)

            # psi_hat should be same format as psi (a batch of render_params)

            # print('im_cube shape', im_cube.shape)

            # getting shapes right for inputs into f_psi:
            # psis shape: torch.Size([4, 10, 9])
            # flattened batch of psis torch.Size([360])
            # batch of styles torch.Size([40, 10])
            # im_cube shape torch.Size([4, 3, 392, 392])

            # f_psi will take a batch of 10 styles and a batch of 1 data cube
            # and return a batch of 10 psis
            # print(im_cube, batch_of_styles)
            
            ############################################
            # choose a batch of styles to pass to f_psi
            # by picking one from each sub-batch 
            perm = torch.randint(f_psi.num_camera_samples, (im_cube.size(0),))
            for j in range(perm.size(0)):
                perm[j] += j*f_psi.num_camera_samples
            small_batch_of_styles = batch_of_styles[perm]
        
            psi_hat = f_psi(im_cube, small_batch_of_styles)
            loss_psi = loss_fn(psi_hat, psi)
            ###########################################

            loss_style = torch.zeros(1).cuda()

            for i, s in enumerate(batch_of_styles):
                # get all ids that are the same as i

                p = i // trainset.camera_samples
                current_im_id = im_2d_cube_id[p][0]
                list_of_same_ids = (batch_of_ids == current_im_id).nonzero().flatten()

                # print(list_of_same_ids)
                # pick one
                index_with_id_same = randomly_choose(list_of_same_ids)
                # get all ids that are different than i
                list_of_different_ids = (
                    (batch_of_ids != current_im_id).nonzero().flatten()
                )
                # print('list of different', list_of_different_ids)

                # pick one
                index_with_id_different = randomly_choose(list_of_different_ids)
                # compute loss

                # TODO: this is differencing Tensors of length 10
                # and instead should do some kind of (Euclidean?) distance that gives a scalar number
                loss_style = (
                    loss_style
                    + torch.dist(s, batch_of_styles[index_with_id_same]) ** 2
                    - torch.dist(s, batch_of_styles[index_with_id_different]) ** 2
                )
            print("new total loss style", loss_style)
            loss_style = loss_style / len(batch_of_styles)
            print("new average loss style", loss_style)
            total_loss = loss_psi + regularization_rate*loss_style
            print("new average loss psi", loss_psi)
            print("new average loss total", total_loss)

            total_loss.backward()

            print(loss_psi, loss_style)

            optimizer.step()

            # print statistics
            running_loss += total_loss.item()
            everyN = 10
            if i % everyN == everyN - 1:  # print every 10 mini-batches
                print(
                    "[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / everyN)
                )
                running_loss = 0.0

    print("Finished Training")


# TODO after running main

# test trained model:

# dataiter = iter(testloader)
# # grab batch of four images again but from data it hasn't seen before
# images, labels = dataiter.next()

# # print images
# imshow(torchvision.utils.make_grid(images))
# print("GroundTruth: ", " ".join("%5s" % classes[labels[j]] for j in range(4)))

# # calc outputs with our trained network
# # outputs are batch size (4) by 10 (number of classes)
# # each number is the confidence that the current image is in that class
# # higher number means more confident
# outputs = net(images)
# # print(outputs)
# # outputs are tensors, but we want the one it thinks is most likely
# # torch.max returns (max_value, index)
# # therefore "predicted" will hold the index of the max value so we can get the name of the class from the classes array
# _, predicted = torch.max(outputs, 1)

# print("Predicted: ", " ".join("%5s" % classes[predicted[j]] for j in range(4)))

# correct = 0
# total = 0
# # calc percentage calc labels match the real labels in the testing set
# with torch.no_grad():  # just in prediction mode, not learning mode
#     for data in testloader:
#         images, labels = data
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(
#     "Accuracy of the network on the 10000 test images: %d %%" % (100 * correct / total)
# )

# class_correct = list(0.0 for i in range(10))
# class_total = list(0.0 for i in range(10))
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         outputs = net(images)
#         _, predicted = torch.max(outputs, 1)
#         c = (predicted == labels).squeeze()
#         for i in range(4):
#             label = labels[i]
#             class_correct[label] += c[i].item()
#             class_total[label] += 1

# for i in range(10):
#     print(
#         "Accuracy of %5s : %2d %%"
#         % (classes[i], 100 * class_correct[i] / class_total[i])
#     )

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # Assuming that we are on a CUDA machine, this should print a CUDA device:

# print(device)
# net.to(device)
# inputs, labels = data[0].to(device), data[1].to(device)

if __name__ == "__main__":
    main()

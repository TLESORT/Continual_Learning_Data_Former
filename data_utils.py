import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

import torch
import os
from torchvision import datasets, transforms

import numpy as np
import imageio

if os.path.exists("dataset_loaders"):
    from dataset_loaders.LSUN import load_LSUN
    from dataset_loaders.cifar10 import load_Cifar10
    from dataset_loaders.cifar100 import load_Cifar100
    from dataset_loaders.core50 import load_core50
    from dataset_loaders.fashion import Fashion
    from dataset_loaders.kmnist import Kmnist
else:
    from .dataset_loaders.LSUN import load_LSUN
    from .dataset_loaders.cifar10 import load_Cifar10
    from .dataset_loaders.cifar100 import load_Cifar100
    from .dataset_loaders.core50 import load_core50
    from .dataset_loaders.fashion import Fashion
    from .dataset_loaders.kmnist import Kmnist

def get_images_format(dataset):

    if dataset == 'MNIST' or dataset == 'fashion' or dataset == 'mnishion' or "mnist" in dataset:
        imageSize = 28
        img_channels = 1
    elif dataset == 'cifar10' or dataset == 'cifar100':
        imageSize = 32
        img_channels = 3
    elif dataset == 'core10' or dataset == 'core50':
        # if args.imageSize is at default value we change it to 128
        imageSize = 128
        img_channels = 3
    else:
        raise Exception("[!] There is no option for " + dataset)

    return imageSize, img_channels


def check_args(args):


    if "mnist_fellowship" in args.task:
        args.dataset = "mnist_fellowship"
        if 'merge' in args.task:
            args.dataset = "mnist_fellowship_merge"

    return args


def check_and_Download_data(folder, dataset, task):
    # download data if possible
    if dataset == 'MNIST' or dataset == 'mnishion' or "mnist_fellowship" in task:
        datasets.MNIST(os.path.join(folder, "MNIST"), train=True, download=True, transform=transforms.ToTensor())
    if dataset == 'fashion' or dataset == 'mnishion' or "mnist_fellowship" in task:
        Fashion(os.path.join(folder, "fashion"), train=True, download=True, transform=transforms.ToTensor())
    # download data if possible
    if dataset == 'kmnist' or "mnist_fellowship" in task:
        Kmnist(os.path.join(folder, "kmnist"), train=True, download=True, transform=transforms.ToTensor())
    if dataset == 'core50' or dataset == 'core10':
        if not os.path.isdir(folder):
            print('This dataset should be downloaded manually')

def load_data(dataset, path2data, train=True):
    if dataset == 'cifar10':
        path2data = os.path.join(path2data, dataset, "processed")
        x_, y_ = load_Cifar10(path2data, train)

        x_ = x_.float()
    elif dataset == 'cifar100':
        path2data = os.path.join(path2data, dataset, "processed")
        x_, y_ = load_Cifar100(path2data, train)

        x_ = x_.float()
    elif dataset == 'LSUN':
        x_, y_ = load_LSUN(path2data, train)

        x_ = x_.float()
    elif dataset == 'core50' or dataset == 'core10':

        x_, y_ = load_core50(dataset, path2data, train)

    elif 'mnist_fellowship' in dataset:
        # In this case data will be loaded later dataset by dataset
        return None, None
    else:

        if train:
            data_file = os.path.join(path2data, dataset, "processed", 'training.pt')
        else:
            data_file = os.path.join(path2data, dataset, "processed", 'test.pt')

        if not os.path.isfile(data_file):
            raise AssertionError("Missing file: {}".format(data_file))

        x_, y_ = torch.load(data_file)
        x_ = x_.float() / 255.0

    y_ = y_.view(-1).long()

    return x_, y_


def visualize_batch(batch, number, shape, path):
    batch = batch.cpu().data

    image_frame_dim = int(np.floor(np.sqrt(number)))

    if shape[2] == 1:
        data_np = batch.numpy().reshape(number, shape[0], shape[1], shape[2])
        save_images(data_np[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    path)
    elif shape[2] == 3:
        data = batch.numpy().reshape(number, shape[2], shape[1], shape[0])
        make_samples_batche(data[:number], number, path)
    else:
        save_images(batch[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    path)


def save_images(images, size, image_path):
    return imsave(images, size, image_path)


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    image -= np.min(image)
    image /= np.max(image) + 1e-12
    image = 255 * image  # Now scale by 255
    image = image.astype(np.uint8)
    return imageio.imwrite(path, image)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')


def img_stretch(img):
    img = img.astype(float)
    img -= np.min(img)
    img /= np.max(img) + 1e-12
    return img


def make_samples_batche(prediction, batch_size, filename_dest):
    plt.figure()
    batch_size_sqrt = int(np.sqrt(batch_size))
    input_channel = prediction[0].shape[0]
    input_dim = prediction[0].shape[1]
    prediction = np.clip(prediction, 0, 1)
    pred = np.rollaxis(prediction.reshape((batch_size_sqrt, batch_size_sqrt, input_channel, input_dim, input_dim)), 2,
                       5)
    pred = pred.swapaxes(2, 1)
    pred = pred.reshape((batch_size_sqrt * input_dim, batch_size_sqrt * input_dim, input_channel))
    fig, ax = plt.subplots(figsize=(batch_size_sqrt, batch_size_sqrt))
    ax.axis('off')
    ax.imshow(img_stretch(pred), interpolation='nearest')
    ax.grid()
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(filename_dest, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    plt.close()

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
    from dataset_loaders.core50 import load_core50
    from dataset_loaders.fashion import Fashion
    from dataset_loaders.kmnist import Kmnist
else:
    from .dataset_loaders.LSUN import load_LSUN
    from .dataset_loaders.cifar10 import load_Cifar10
    from .dataset_loaders.core50 import load_core50
    from .dataset_loaders.fashion import Fashion
    from .dataset_loaders.kmnist import Kmnist



def check_args(args):
    if args.dataset == 'MNIST' or args.dataset == 'fashion' or args.dataset == 'kmnist' or args.dataset == 'mnishion' or args.task == "mnist_fellowship":
        args.imageSize = 28
        args.img_channels = 1
    elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
        args.imageSize = 32
        args.img_channels = 3
    elif args.dataset == 'core10' or args.dataset == 'core50':
        # if args.imageSize is at default value we change it to 128
        if args.imageSize == 28:
            args.imageSize = 128
        args.img_channels = 3
    else:
        raise Exception("[!] There is no option for " + args.dataset)

    if args.task == "mnist_fellowship":
        args.dataset = "mnist_fellowship"

    return args


def check_and_Download_data(folder, dataset, task):
    # download data if possible
    if dataset == 'MNIST' or dataset == 'mnishion' or task == "mnist_fellowship":
        datasets.MNIST(folder, train=True, download=True, transform=transforms.ToTensor())
    if dataset == 'fashion' or dataset == 'mnishion' or task == "mnist_fellowship":
        Fashion(os.path.join(folder, "fashion"), train=True, download=True, transform=transforms.ToTensor())
    # download data if possible
    if dataset == 'kmnist' or task == "mnist_fellowship":
        Kmnist(os.path.join(folder, "kmnist"), train=True, download=True, transform=transforms.ToTensor())
    if dataset == 'core50' or dataset == 'core10':
        if not os.path.isdir(folder):
            print('This dataset should be downloaded manually')

def load_data(dataset, path2data, imageSize=32, path_only=False):
    if dataset == 'cifar10':
        path2data = os.path.join(path2data, dataset, "processed")
        x_tr, y_tr, x_te, y_te = load_Cifar10(path2data)

        x_tr = x_tr.float()
        x_te = x_te.float()

    elif dataset == 'LSUN':
        x_tr, y_tr, x_te, y_te = load_LSUN(path2data)

        x_tr = x_tr.float()
        x_te = x_te.float()
    elif dataset == 'core50' or dataset == 'core10':

        x_tr, y_tr, x_te, y_te = load_core50(dataset, path2data, imageSize=imageSize, path_only=path_only)

    elif dataset == 'mnist_fellowship':
        # In this case data will be loaded later dataset by dataset
        return None, None, None, None
    else:

        train_file = os.path.join(path2data, dataset, "processed", 'training.pt')
        test_file = os.path.join(path2data, dataset, "processed", 'test.pt')

        if not os.path.isfile(train_file):
            raise AssertionError("Missing file: {}".format(train_file))

        if not os.path.isfile(test_file):
            raise AssertionError("Missing file: {}".format(test_file))

        x_tr, y_tr = torch.load(train_file)
        x_te, y_te = torch.load(test_file)

        x_tr = x_tr.float() / 255.0
        x_te = x_te.float() / 255.0

    y_tr = y_tr.view(-1).long()
    y_te = y_te.view(-1).long()

    return x_tr, y_tr, x_te, y_te


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

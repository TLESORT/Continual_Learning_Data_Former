import os.path
import torch
import numpy as np


def load_data(path, imageSize, path_only=False):
    if path_only:
        return load_path(path)
    else:
        return load_both(path, imageSize)


def load_path(path):
    path_tr = np.load(path)['paths']
    y_tr = np.load(path)['y']
    y_tr = y_tr.reshape((-1))
    y_tr = torch.Tensor(y_tr)
    return path_tr, y_tr


def load_both(path, imageSize):
    x_tr = np.load(path)['x']
    y_tr = np.load(path)['y']

    x_tr = x_tr.reshape((-1, imageSize, imageSize, 3))
    y_tr = y_tr.reshape((-1))

    if not x_tr.shape[0] == y_tr.shape[0]:
        raise AssertionError("There is something wrong here")

    ## AS FOR CIFAR10 WE DO mean 0.5 and std 0.5

    stds = x_tr.std((0, 1, 2))

    ## *2 for 0.5 std
    x_tr[:, :, :, 0] /= 2 * stds[0]
    x_tr[:, :, :, 1] /= 2 * stds[1]
    x_tr[:, :, :, 2] /= 2 * stds[2]

    means = x_tr.mean((0, 1, 2))

    ## - -0.5 (  <=> +0.5 ) for 0.5 mean
    x_tr[:, :, :, 0] -= means[0] - 0.5
    x_tr[:, :, :, 1] -= means[1] - 0.5
    x_tr[:, :, :, 2] -= means[2] - 0.5

    x_tr = np.transpose(x_tr, (0, 3, 1, 2))

    x_tr = torch.Tensor(x_tr)
    y_tr = torch.Tensor(y_tr)

    return x_tr, y_tr


def load_core50(path, imageSize=32, path_only=False):
    if path_only:
        path_train = os.path.join(path, 'core50_paths_train.npz')
        path_test = os.path.join(path, 'core50_paths_test.npz')
    else:
        path_train = os.path.join(path, 'core50_imgs_' + str(imageSize) + '_train.npz')
        path_test = os.path.join(path, 'core50_imgs_' + str(imageSize) + '_test.npz')

    x_tr, y_tr = load_data(path_train, imageSize, path_only)
    x_te, y_te = load_data(path_test, imageSize, path_only)

    return x_tr, y_tr, x_te, y_te

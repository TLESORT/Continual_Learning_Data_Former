
import torch
import pytest
import os

from builders.disjoint import Disjoint
from builders.mnistfellowship import MnistFellowship

dir_data = "Archives"
dir_samples = "Samples"

# command MNIST : python main.py --task disjoint --n_tasks 10 --dataset MNIST
# command fashion : python main.py --task disjoint --n_tasks 10 --dataset fashion
@pytest.mark.parametrize("dataset", ["MNIST", "fashion", "kmnist"])
@pytest.mark.parametrize("ind_task", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_disjoint_samples_train_10_tasks(dataset, ind_task):
    input_folder = os.path.join(dir_data, 'Data')
    data_set = Disjoint(path=input_folder, dataset=dataset, tasks_number=10, download=False, train=True)

    path = os.path.join(dir_data, "Data", "Continua", dataset, "Disjoint_10_train.pt")
    if not os.path.isfile(path):
        raise Exception("[!] file '{}' does not exists " + path)

    data_set.set_task(ind_task)

    folder = os.path.join(dir_samples, "disjoint_10_tasks")

    if not os.path.exists(folder):
        os.makedirs(folder)

    path_out = os.path.join(folder, "{}_task_{}.png".format(dataset, ind_task))
    data_set.visualize_sample(path_out, number=100, shape=[28, 28, 1])

@pytest.mark.parametrize("dataset", ["MNIST"])
@pytest.mark.parametrize("ind_task", [0, 1, 2, 3, 4])
def test_disjoint_samples_train_5_tasks(dataset, ind_task):

    input_folder = os.path.join(dir_data, 'Data')
    data_set = Disjoint(path=input_folder, dataset=dataset, tasks_number=5, download=False, train=True)

    path = os.path.join(dir_data, "Data", "Continua", dataset, "Disjoint_5_train.pt")
    if not os.path.isfile(path):
        raise Exception("[!] file '{}' does not exists " + path)

    data_set.set_task(ind_task)

    folder = os.path.join(dir_samples, "disjoint_5_tasks")

    if not os.path.exists(folder):
        os.makedirs(folder)

    path_out = os.path.join(folder, "{}_task_{}.png".format(dataset, ind_task))
    data_set.visualize_sample(path_out, number=100, shape=[28, 28, 1])


# command : python main.py --task mnist_fellowship --n_tasks 3
@pytest.mark.parametrize("ind_task", [0, 1, 2])
def test_samples_mnist_fellowship(ind_task):
    input_folder = os.path.join(dir_data, 'Data')
    data_set = MnistFellowship(path=input_folder, tasks_number=3, download=False, train=True)

    path = os.path.join(dir_data, "Data", "Continua", "mnist_fellowship", "mnist_fellowship_3_train.pt")
    if not os.path.isfile(path):
        raise Exception("[!] file '{}' does not exists " + path)

    data_set.set_task(ind_task)
    folder = os.path.join(dir_samples, "mnist_fellowship")

    if not os.path.exists(folder):
        os.makedirs(folder)

    path_out = os.path.join(folder, "mnist_fellowship_task_{}.png".format(ind_task))

    data_set.visualize_sample(path_out, number=100, shape=[28, 28, 1])


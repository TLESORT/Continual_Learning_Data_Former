from data_loader import DatasetLoader
import torch
import pytest
import os


# command MNIST : python main.py --task disjoint --n_tasks 10 --dataset MNIST
# command fashion : python main.py --task disjoint --n_tasks 10 --dataset fashion
@pytest.mark.parametrize("dataset", ["MNIST", "fashion", "kmnist"])
@pytest.mark.parametrize("ind_task", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_disjoint_samples_train_10_tasks(dataset, ind_task):
    path = "./Archives/Data/Tasks/{}/disjoint_10_train.pt".format(dataset)
    data = torch.load(path)
    data_set = DatasetLoader(data, current_task=0, transform=None, load_images=False, path=None)

    data_set.set_task(ind_task)

    folder = "./Samples/disjoint_10_tasks/"

    if not os.path.exists(folder):
        os.makedirs(folder)

    path_out = os.path.join(folder, "{}_task_{}.png".format(dataset, ind_task))
    data_set.visualize_sample(path_out, number=100, shape=[28, 28, 1])


# @pytest.mark.parametrize("task", ["permutations", "rotations"])
# command rotation : python main.py --task rotations --n_tasks 5 --dataset MNIST
# command permutations : python main.py --task permutations --n_tasks 5 --dataset MNIST
@pytest.mark.parametrize("task", ["permutations", "rotations"])
@pytest.mark.parametrize("dataset", ["MNIST"])
@pytest.mark.parametrize("ind_task", [0, 1, 2, 3, 4])
def test_disjoint_samples_train_5_tasks(task, dataset, ind_task):
    path = "./Archives/Data/Tasks/{}/{}_5_train.pt".format(dataset, task)
    data = torch.load(path)
    data_set = DatasetLoader(data, current_task=0, transform=None, load_images=False, path=None)

    data_set.set_task(ind_task)
    folder = "./Samples/5_tasks/"

    if not os.path.exists(folder):
        os.makedirs(folder)

    path_out = os.path.join(folder, "{}_{}_task_{}.png".format(dataset, task, ind_task))

    if task == "permutations":
        permutations = torch.load("./Archives/Data/Tasks/{}/ind_permutations_5_train.pt".format(dataset))
        data_set.visualize_reordered(path_out, number=100, shape=[28, 28, 1], permutations=permutations)
    else:
        data_set.visualize_sample(path_out, number=100, shape=[28, 28, 1])


# command : python main.py --task mnist_fellowship --n_tasks 3
@pytest.mark.parametrize("ind_task", [0, 1, 2])
def test_disjoint_samples_mnist_fellowship(ind_task):
    path = "./Archives/Data/Tasks/mnist_fellowship/mnist_fellowship_3_train.pt"
    data = torch.load(path)
    data_set = DatasetLoader(data, current_task=0, transform=None, load_images=False, path=None)

    data_set.set_task(ind_task)
    folder = "./Samples/mnist_fellowship/"

    if not os.path.exists(folder):
        os.makedirs(folder)

    path_out = os.path.join(folder, "mnist_fellowship_task_{}.png".format(ind_task))

    data_set.visualize_sample(path_out, number=100, shape=[28, 28, 1])


# command : python main.py --task disjoint_classes_permutations --n_tasks 10 --index_permutation 2 --dataset MNIST
@pytest.mark.parametrize("dataset", ["MNIST"])
@pytest.mark.parametrize("ind_task", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_disjoint_samples_disjoint_classes_permutations(ind_task, dataset):
    index_permutation = 2  # empirically chosen
    permutation = torch.load("permutation_classes.t")[index_permutation]

    name = ''
    for i in range(10):
        name += str(int(permutation[i]))

    path = "./Archives/Data/Tasks/{}/disjoint_{}_10_train.pt".format(dataset, name)
    data = torch.load(path)
    data_set = DatasetLoader(data, current_task=0, transform=None, load_images=False, path=None)

    data_set.set_task(ind_task)
    folder = "./Samples/disjoint_classes_permutations/"

    if not os.path.exists(folder):
        os.makedirs(folder)

    path_out = os.path.join(folder, "disjoint_classes_permutations_{}.png".format(ind_task))

    data_set.visualize_sample(path_out, number=100, shape=[28, 28, 1])

# command : python main.py --task disjoint_rotations --n_tasks 30 --dataset MNIST
@pytest.mark.hello
@pytest.mark.parametrize("dataset", ["MNIST"])
@pytest.mark.parametrize("ind_task",
                         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                          26, 27, 28, 29])
def test_disjoint_samples_disjoint_rotation(ind_task, dataset):
    path = "./Archives/Data/Tasks/{}/disjoint_rotations_30_train.pt".format(dataset)
    data = torch.load(path)
    data_set = DatasetLoader(data, current_task=0, transform=None, load_images=False, path=None)

    data_set.set_task(ind_task)

    folder = "./Samples/disjoint_rotations_30/"

    if not os.path.exists(folder):
        os.makedirs(folder)

    path_out = os.path.join(folder, "{}_task_{}.png".format(dataset, ind_task))
    data_set.visualize_sample(path_out, number=100, shape=[28, 28, 1])

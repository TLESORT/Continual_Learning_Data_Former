import pytest
from tests.utils_tests import check_task_sequences_files
from builders.disjoint import Disjoint
import os

dataset_size = 100
dir_data = "./Archives"


#@pytest.mark.parametrize("datasets", ["mnist", "fashion", "kmnist","cifar10","LSUN","core50"])
@pytest.mark.skip(reason="Too memory hungry")
@pytest.mark.parametrize("dataset", ["MNIST", "fashion", "kmnist", "cifar10"])
def test_download(tmpdir, dataset):
    continuum = Disjoint(path=tmpdir, dataset=dataset, tasks_number=1, download=False, train=True)

    if continuum is None:
        raise AssertionError("Object construction has failed")

# #@pytest.mark.parametrize("datasets", ["mnist", "fashion", "kmnist","cifar10","LSUN","core50"])
@pytest.mark.slow
@pytest.mark.parametrize("dataset", ["MNIST", "fashion", "kmnist"])
@pytest.mark.parametrize("n_tasks", [1, 5, 10])
def test_disjoint_vanilla_train(dataset, n_tasks):
    # no need to download the dataset again for this test (if it already exists)
    input_folder = os.path.join(dir_data, 'Data')
    Disjoint(path=input_folder, dataset=dataset, tasks_number=n_tasks, download=False, train=True)
    check_task_sequences_files(scenario="Disjoint", folder=dir_data, n_tasks=n_tasks, dataset=dataset, train=True)

@pytest.mark.slow
@pytest.mark.parametrize("dataset", ["MNIST", "fashion", "kmnist"])
@pytest.mark.parametrize("n_tasks", [1, 5, 10])
def test_disjoint_vanilla_test(dataset, n_tasks):
    # no need to download the dataset again for this test (if it already exists)
    input_folder = os.path.join(dir_data, 'Data')
    Disjoint(path=input_folder, dataset=dataset, tasks_number=n_tasks, download=False, train=False)
    check_task_sequences_files(scenario="Disjoint", folder=dir_data, n_tasks=n_tasks, dataset=dataset, train=False)



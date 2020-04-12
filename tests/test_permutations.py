import pytest
from tests.utils_tests import check_task_sequences_files
from continuum.permutations import Permutations
import os

dataset_size = 100
dir_data = "./Archives"


# #@pytest.mark.parametrize("datasets", ["mnist", "fashion", "kmnist","cifar10","LSUN","core50"])
@pytest.mark.slow
@pytest.mark.parametrize("dataset", ["MNIST"])
@pytest.mark.parametrize("n_tasks", [3, 5])
def test_permutations_train(dataset, n_tasks):
    # no need to download the dataset again for this test (if it already exists)
    input_folder = os.path.join(dir_data, 'Data')
    Permutations(path=input_folder, dataset=dataset, tasks_number=n_tasks, download=False, train=True)
    check_task_sequences_files(scenario="Rotations", folder=dir_data, n_tasks=n_tasks, dataset=dataset, train=True)
import pytest
from tests.utils_tests import check_task_sequences_files
from continuum.mnistfellowship import MnistFellowship
import os

dataset_size = 100
dir_data = "./Archives"

@pytest.mark.slow
def test_mnist_fellowship():
    # no need to download the dataset again for this test (if it already exists)
    input_folder = os.path.join(dir_data, 'Data')
    MnistFellowship(path=input_folder, merge=False, download=False, train=True)
    check_task_sequences_files(scenario="mnist_fellowship",
                               folder=dir_data,
                               n_tasks=3,
                               dataset="mnist_fellowship",
                               train=True)

@pytest.mark.slow
def test_mnist_fellowship_merge():
    # no need to download the dataset again for this test (if it already exists)
    input_folder = os.path.join(dir_data, 'Data')
    MnistFellowship(path=input_folder, merge=True, download=False, train=True)
    check_task_sequences_files(scenario="mnist_fellowship_merge",
                               folder=dir_data,
                               n_tasks=3,
                               dataset="mnist_fellowship",
                               train=True)
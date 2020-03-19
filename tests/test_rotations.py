import pytest
from tests.utils_tests import check_task_sequences_files
from Sequence_Formers.rotations import Rotations
from Sequence_Formers.disjoint_rotations import DisjointRotations
import os

dataset_size = 100
dir_data = "./Archives"


# #@pytest.mark.parametrize("datasets", ["mnist", "fashion", "kmnist","cifar10","LSUN","core50"])
@pytest.mark.slow
@pytest.mark.parametrize("dataset", ["MNIST"])
@pytest.mark.parametrize("n_tasks", [3])
@pytest.mark.parametrize("rotation_min", [0.0, 45.0])
@pytest.mark.parametrize("rotation_max", [90.0, 180.0])
def test_rotations_train(dataset, n_tasks, rotation_min, rotation_max):
    # no need to download the dataset again for this test (if it already exists)
    input_folder = os.path.join(dir_data, 'Data')
    Data_Former = Rotations(path=input_folder, dataset=dataset, tasks_number=n_tasks, download=False, train=True,
                            min_rot=rotation_min,
                            max_rot=rotation_max)
    check_task_sequences_files(scenario="Rotations", folder=dir_data, n_tasks=n_tasks, dataset=dataset, train=True)
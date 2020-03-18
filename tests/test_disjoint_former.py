import pytest
from tests.utils_tests import check_task_sequences_files
from Sequence_Formers.disjoint import Disjoint
from Sequence_Formers.disjoint_rotations import DisjointRotations
import os

dataset_size = 100
dir_data = "."




@pytest.fixture
def get_args():
    return args()



#@pytest.mark.parametrize("datasets", ["mnist", "fashion", "kmnist","cifar10","LSUN","core50"])
@pytest.mark.skip(reason="Too memory hungry")
@pytest.mark.parametrize("dataset", ["MNIST", "fashion", "kmnist", "cifar10"])
def test_download(tmpdir, get_args, dataset):
    args = get_args
    args.dir = tmpdir
    args.dataset = dataset
    args.n_tasks = 1
    args.set_paths()
    Data_Former = Disjoint(args)

    if Data_Former is None:
        raise AssertionError("Object construction has failed")

# #@pytest.mark.parametrize("datasets", ["mnist", "fashion", "kmnist","cifar10","LSUN","core50"])
@pytest.mark.slow
@pytest.mark.parametrize("dataset", ["MNIST", "fashion", "kmnist"])
@pytest.mark.parametrize("n_tasks", [1, 5, 10])
def test_disjoint_vanilla_train(dataset, n_tasks):
    # no need to download the dataset again for this test (if it already exists)
    input_folder = os.path.join(dir_data, 'Data')
    Data_Former = Disjoint(path=input_folder, dataset=dataset, tasks_number=n_tasks, download=False, train=True)
    check_task_sequences_files(scenario="Disjoint", folder=dir_data, n_tasks=n_tasks, dataset=dataset, train=True)

@pytest.mark.slow
@pytest.mark.parametrize("dataset", ["MNIST", "fashion", "kmnist"])
@pytest.mark.parametrize("n_tasks", [1, 5, 10])
def test_disjoint_vanilla_test(dataset, n_tasks):
    # no need to download the dataset again for this test (if it already exists)
    input_folder = os.path.join(dir_data, 'Data')
    Data_Former = Disjoint(path=input_folder, dataset=dataset, tasks_number=n_tasks, download=False, train=False)
    check_task_sequences_files(scenario="Disjoint", folder=dir_data, n_tasks=n_tasks, dataset=dataset, train=False)


# #@pytest.mark.parametrize("datasets", ["mnist", "fashion", "kmnist","cifar10","LSUN","core50"])
@pytest.mark.skip(reason="not yet implemented")
@pytest.mark.slow
@pytest.mark.parametrize("dataset", ["MNIST"])
@pytest.mark.parametrize("n_tasks", [30])
def test_disjoint_rotations(tmpdir, get_args, dataset, n_tasks):
    args = get_args
    args.add_supp_parameters("Disjoint_rotation")
    args.dir = tmpdir
    args.dataset = dataset
    args.n_tasks = n_tasks
    args.task = "disjoint_rotation"
    args.set_paths()
    # no need to download the dataset again for this test (if it already exists)
    args.i = os.path.join(dir_archive, 'Data', 'Datasets')
    Data_Former = DisjointRotations(args)
    Data_Former.formating_data()

    check_task_sequences_files(args.task, tmpdir, n_tasks, dataset)



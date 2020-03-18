import pytest
from Sequence_Formers.disjoint import Disjoint
from Sequence_Formers.rotations import Rotations
from Sequence_Formers.disjoint_rotations import DisjointRotations
import os

dataset_size = 100
dir_data = "."


class args(object):
    def __init__(self):
        self.n_tasks = 10
        self.num_classes = 10
        self.task = "disjoint"
        self.img_channels = 1
        self.imageSize = 28
        self.dir = None
        self.dataset = None
        self.path_only = None
        self.verbose = False

    def set_paths(self):

        if self.dir is None:
            raise AssertionError("Dir is None")
        if self.dataset is None:
            raise AssertionError("dataset is None")
        self.o = os.path.join(self.dir, 'Data', 'Continua', self.dataset)
        self.i = os.path.join(self.dir, 'Data', 'Datasets')

    def add_supp_parameters(self, method):

        if "rotation" in method:
            self.min_rot = 0.0
            self.max_rot = 90.0


@pytest.fixture
def get_args():
    return args()

def check_task_sequences_files(scenario, folder,n_tasks, dataset, train=True):

    if train:
        filename = "{}_{}_{}.pt".format(scenario, n_tasks, "train")
    else:
        filename = "{}_{}_{}.pt".format(scenario, n_tasks, "test")

    path = os.path.join(folder, "Data", "Continua", dataset, filename)
    if not os.path.isfile(path):
        raise AssertionError("Test fail with file : {}".format(path))



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



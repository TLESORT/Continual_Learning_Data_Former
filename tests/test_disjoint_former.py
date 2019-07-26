
import pytest
from ..Sequence_Formers.disjoint import Disjoint
import os

dataset_size = 100
dir_archive = "../Archives"


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

    def set_paths(self):
        assert self.dir is not None
        assert self.dataset is not None
        self.o = os.path.join(self.dir,'Data','Tasks',self.dataset)
        self.i = os.path.join(self.dir,'Data','Datasets')


    def add_supp_parameters(self, method):
        pass

@pytest.fixture
def get_args():
    return args()

# #@pytest.mark.parametrize("datasets", ["mnist", "fashion", "kmnist","cifar10","LSUN","core50"])
@pytest.mark.parametrize("dataset", ["MNIST", "fashion", "kmnist", "cifar10"])
def test_download(tmpdir, get_args, dataset):
    args = get_args
    args.dir = tmpdir
    args.dataset = dataset
    args.n_tasks = 1
    args.set_paths()
    Data_Former = Disjoint(args)


# #@pytest.mark.parametrize("datasets", ["mnist", "fashion", "kmnist","cifar10","LSUN","core50"])
@pytest.mark.parametrize("dataset", ["MNIST", "fashion", "kmnist", "cifar10"])
@pytest.mark.parametrize("n_tasks", [1, 5, 10])
def test_disjoint_vanilla(tmpdir, get_args, dataset, n_tasks):
    args = get_args
    args.dir = tmpdir
    args.dataset = dataset
    args.n_tasks = n_tasks
    args.set_paths()
    # no need to download the dataset again for this test (if it already exists)
    args.i = os.path.join(dir_archive, 'Data', 'Datasets')
    Data_Former = Disjoint(args)
    Data_Former.formating_data()

    filename =  "disjoint_{}_{}.pt".format(n_tasks,"train")

    if not os.path.isfile(os.path.join(tmpdir,"Data","Tasks",dataset,filename)):
        raise AssertionError("Test fail with train file")

    filename =  "disjoint_{}_{}.pt".format(n_tasks,"valid")
    if not os.path.isfile(os.path.join(tmpdir,"Data","Tasks",dataset,filename)):
        raise AssertionError("Test fail with valid file")

    filename =  "disjoint_{}_{}.pt".format(n_tasks,"test")
    if not os.path.isfile(os.path.join(tmpdir,"Data","Tasks",dataset,filename)):
        raise AssertionError("Test fail with test file")
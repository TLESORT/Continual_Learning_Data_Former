from ..data_loader import DatasetLoader
import torch
import pytest
from torch.utils import data

dataset_size = 100

@pytest.fixture
def get_fake_dataset():
    liste_label = torch.randperm(dataset_size).long()
    data = torch.rand([dataset_size, 3, 28, 28])

    fake_dataset = []
    fake_dataset.append([(0, 10), data, liste_label])
    fake_dataset.append([(0, 10), data, liste_label])
    fake_dataset.append([(0, 10), data, liste_label])

    return fake_dataset


def test_DataLoader_init(get_fake_dataset):
    fake_dataset = get_fake_dataset
    dataset = DatasetLoader(fake_dataset)

    if not dataset.current_task == 0:
        raise AssertionError("Test fail")


def test_DataLoader_init_label_is_dict(get_fake_dataset):
    """
    Test if the dictionnary of label is really a dictionnary
    :param get_fake_dataset:
    :return:
    """
    fake_dataset = get_fake_dataset
    dataset = DatasetLoader(fake_dataset)

    if not isinstance(dataset.labels, dict):
        raise AssertionError("Test fail")

def test_DataLoader_init_label_size(get_fake_dataset):
    """
    Test if the dictionnary of label have the good size
    :param get_fake_dataset:
    :return:
    """
    fake_dataset = get_fake_dataset
    dataset = DatasetLoader(fake_dataset)

    if not len(dataset.labels) == dataset_size:
        raise AssertionError("Test fail")



@pytest.mark.parametrize("init_current_task", [0, 1, 2])
def test_DataLoader_init_current_task(get_fake_dataset, init_current_task):
    fake_dataset = get_fake_dataset
    dataset = DatasetLoader(fake_dataset, current_task=init_current_task)

    if not dataset.current_task == init_current_task:
        raise AssertionError("Test fail")


def test_DataLoader_with_torch(get_fake_dataset):
    """
    Test if the dataloader can be used with torch.utils.data.DataLoader
    :param get_fake_dataset:
    :return:
    """
    fake_dataset = get_fake_dataset
    dataset = DatasetLoader(fake_dataset)
    train_loader = data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=6)

    for t, (batch, label) in enumerate(train_loader):
        break

def test_DataLoader_with_torch_loader(get_fake_dataset):
    """
    Test if the dataloader with torch.utils.data.DataLoader provide data of good type
    :param get_fake_dataset:
    :return:
    """
    fake_dataset = get_fake_dataset
    dataset = DatasetLoader(fake_dataset)
    train_loader = data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=6)

    for t, (batch, label) in enumerate(train_loader):

        if not isinstance(label, torch.LongTensor):
            raise AssertionError("Test fail")

        if not isinstance(batch, torch.FloatTensor):
            raise AssertionError("Test fail")
        break

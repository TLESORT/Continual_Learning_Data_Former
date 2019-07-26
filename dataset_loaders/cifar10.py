
import torch
from torchvision import datasets, transforms


def load_Cifar10(path):
    trans = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset_train = datasets.CIFAR10(root=path, train=True, download=True, transform=trans)
    tensor_data = torch.Tensor(len(dataset_train), 3, 32, 32)

    tensor_label = torch.LongTensor(len(dataset_train))

    for i, (data, label) in enumerate(dataset_train):
        tensor_data[i] = data
        tensor_label[i] = label

    dataset_test = datasets.CIFAR10(root=path, train=False, download=True, transform=trans)

    tensor_test = torch.Tensor(len(dataset_test), 3, 32, 32)
    tensor_label_test = torch.LongTensor(len(dataset_test))

    for i, (data, label) in range(len(dataset_test)):
        tensor_test[i] = data
        tensor_label_test[i] = label

    return tensor_data, tensor_label, tensor_test, tensor_label_test
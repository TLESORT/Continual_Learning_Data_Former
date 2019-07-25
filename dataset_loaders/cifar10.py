
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

    for i in range(len(dataset_train)):
        tensor_data[i] = dataset_train[i][0]
        tensor_label[i] = dataset_train[i][1]

    dataset_test = datasets.CIFAR10(root=path, train=False, download=True, transform=trans)

    tensor_test = torch.Tensor(len(dataset_test), 3, 32, 32)
    tensor_label_test = torch.LongTensor(len(dataset_test))

    for i in range(len(dataset_test)):
        tensor_test[i] = dataset_test[i][0]
        tensor_label_test[i] = dataset_test[i][1]

    return tensor_data, tensor_label, tensor_test, tensor_label_test
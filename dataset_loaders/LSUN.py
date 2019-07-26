import torch
from torchvision import datasets, transforms


def load_LSUN():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset_train = datasets.LSUN(root='/slowdata/LSUN',
                                  classes=['bridge_train', 'church_outdoor_train', 'classroom_train',
                                           'dining_room_train', 'tower_train'], transform=transform)

    dataset_test = datasets.LSUN(root='/slowdata/LSUN',
                                 classes=['bridge_val', 'church_outdoor_val', 'classroom_val',
                                          'dining_room_val', 'tower_val'],
                                 transform=transform)

    data_size = 100000
    test_size = 1000

    tensor_data = torch.Tensor(data_size, 3, 64, 64)
    tensor_label = torch.LongTensor(data_size)

    tensor_test = torch.Tensor(test_size, 3, 64, 64)
    tensor_label_test = torch.LongTensor(test_size)

    for i in range(data_size):
        tensor_data[i] = dataset_train[i][0]
        tensor_label[i] = dataset_train[i][1]

    for i in range(test_size):
        tensor_test[i] = dataset_test[i][0]
        tensor_label_test[i] = dataset_test[i][1]

    return tensor_data, tensor_label, tensor_test, tensor_label_test

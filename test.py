
from data_loader import Dataset_Loader
import os
import torch
from torch.utils import data

path = "./Archives/Data/Tasks/MNIST/disjoint_10_train.pt"
Data = torch.load(path)
data_set = Dataset_Loader(Data, current_task=0, transform=None, load_images=False, path=None)

# Visualize samples for each tasks
for i in range(10):
    data_set.set_task(i)

    folder = "./Samples/disjoint_10_tasks/"

    if not os.path.exists(folder):
        os.makedirs(folder)

    path_samples = os.path.join(folder, "MNIST_task_{}.png".format(i))
    data_set.visualize_sample(path_samples , number=100, shape=[28 ,28 ,1])

# use the dataset with pytorch dataloader for training an algo

# create dataloader
train_loader = data.DataLoader(data_set, batch_size=64, shuffle=True, num_workers=6)

# set the task on 0 for example with the data_set
data_set.set_task(0)

# iterate on task 0
for t, (data, target) in enumerate(train_loader):
    print(target)

# change the task to 2 for example
data_set.set_task(2)

# iterate on task 2
for t, (data, target) in enumerate(train_loader):
    print(target)
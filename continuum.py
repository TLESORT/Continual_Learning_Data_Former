


from builders.disjoint import Disjoint
from builders.mnistfellowship import MnistFellowship
from torch.utils import data


dir_data = "Archives/Data"
dataset= "MNIST"
#continuum = Disjoint(path=dir_data, dataset=dataset, tasks_number=10, download=True, train=True)
continuum = MnistFellowship(path=dir_data, tasks_number=3, download=False, train=True)


train_loader = data.DataLoader(continuum, batch_size=64, shuffle=True, num_workers=6)

for i_, (x_, t_) in enumerate(train_loader):
    if i_ == 0:
        print(x_.shape)
        print(t_)

continuum.set_task(0)
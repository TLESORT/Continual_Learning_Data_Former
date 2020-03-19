


from Sequence_Formers.disjoint import Disjoint
from torch.utils import data

continuum = Disjoint(dataset="MNIST", tasks_number=3)

train_loader = data.DataLoader(continuum, batch_size=64, shuffle=True, num_workers=6)

for i_, (x_, t_) in enumerate(train_loader):
    if i_ == 0:
        print(x_.shape)
        print(t_)

continuum.set_task(0)
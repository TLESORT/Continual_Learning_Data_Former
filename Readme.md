## Continual Data Former

This repositery proprose several script to create sequence of tasks for continual learning

The following type of sequence are possible :

-   Disjoint tasks : each task propose new classes
-   Rotations tasks : each tasks propose same data but with different rotations of datata point
-   Permutations tasks : each tasks propose same data but with different permutations of pixels
-   Mnist Fellowship task : each task is a new mnist like dataset (this sequence of task is an original contribution of this repository)

Several dataset can be used :

-   Mnist
-   fashion-Mnist
-   kmnist
-   cifar10
-   mnishion : concatenation of Mnist and Fashion-Mnist
-   core50 (in developpment)

Some supplementary option are possible:
-   Class order can be shuffled for disjoint tasks
-   We can choose the magnitude of rotation for rotations mnist
-   Of course we can choose the number of tasks (1, 3, 5 and 10 have been tested normally)


### Few possible commands

-   Disjoint tasks

```bash
#MNIST with 10 tasks of one class
python main.py --dataset MNIST --task disjoint --n_tasks 10 --dir ./Archives
```
-   Rotations tasks

```bash
#MNIST with 5 tasks with various rotations
python main.py --dataset MNIST --task rotations --n_tasks 5 --min_rot 0 --max_rot 90 --dir ./Archives
```

-   Permutations tasks

```bash
#MNIST with 10 tasks of one class
python main.py --dataset MNIST --task permutations --n_tasks 5 --dir ./Archives
```

-   Dijsoint_classes_permutations tasks

```bash
#MNIST with 10 tasks of one class
python main.py --dataset MNIST --task dijsoint_classes_permutations --n_tasks 10 --index_permutation 2 --dir ./Archives
```

### Example of use

```python
#MNIST with 10 tasks of one class
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
    data_set.visualize_sample(path_samples , number=100, shape=[28,28,1])
    
# use the dataset with pytorch dataloader for training an algo

# create dataloader
train_loader = data.DataLoader(data_set, batch_size=64, shuffle=True, num_workers=6)

#set the task on 0 for example with the data_set
data_set.set_task(0)

# iterate on task 0
for t, (data, target) in enumerate(train_loader):
    print(target)
    
#change the task to 2 for example
data_set.set_task(2)

# iterate on task 2
for t, (data, target) in enumerate(train_loader):
    print(target)

```

### Citing the Project

```Array.<string>
@misc{continual-learning-data-former,
  author = {Lesort, Timothée},
  title = {Continual Learning Data Former},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/TLESORT/Continual_Learning_Data_Former}},
}

```

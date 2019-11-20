## Continual Learning Data Former

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/9273eb0f97b946308248b0007e054e54)](https://app.codacy.com/app/TLESORT/Continual_Learning_Data_Former?utm_source=github.com&utm_medium=referral&utm_content=TLESORT/Continual_Learning_Data_Former&utm_campaign=Badge_Grade_Dashboard)

This repositery proprose several script to create sequence of tasks for continual learning. The spirit is the following : 
Instead of managing the sequence of tasks while learning, we create the sequence of tasks first and then we load tasks 
one by one while learning.

It makes programming easier and code cleaner.

### Task sequences possibilities

-   **Disjoint tasks** : each task propose new classes
-   **Rotations tasks** : each tasks propose same data but with different rotations of datata point
-   **Permutations tasks** : each tasks propose same data but with different permutations of pixels
-   **Mnist Fellowship task** : each task is a new mnist like dataset (this sequence of task is an original contribution of this repository)

### Example



Just use:

<img src="/Samples/disjoint_5_tasks/MNIST_task_0.png" width="150">  | <img src="/Samples/disjoint_5_tasks/MNIST_task_1.png" width="150">  | <img src="/Samples/disjoint_5_tasks/MNIST_task_2.png" width="150">  | <img src="/Samples/disjoint_5_tasks/MNIST_task_3.png" width="150">  | <img src="/Samples/disjoint_5_tasks/MNIST_task_4.png" width="150">    
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
Task 0 | Task 1 | Task 2 | Task 3 | Task 4


### Datasets

-   Mnist
-   fashion-Mnist
-   kmnist
-   cifar10
-   mnishion : concatenation of Mnist and Fashion-Mnist
-   Core50/Core10

### Some supplementary option are possible
-   The number of tasks can be choosed (1, 3, 5 and 10 have been tested normally)
-   Classes order can be shuffled for disjoint tasks
-   We can choose the magnitude of rotation for rotations mnist

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

First we create the sequence of tasks and save it
```bash
#MNIST with 10 tasks of one class
python main.py --dataset MNIST --task disjoint --n_tasks 10 --dir ./Archives
```

Then we can use the saved sequence in another program for continual learning
```python
#MNIST with 10 tasks of one class
from data_loader import DatasetLoader
import os
import torch
from torch.utils import data

# Path to the task sequence
path = "./Archives/Data/Tasks/MNIST/disjoint_10_train.pt"
# load the file
Data = torch.load(path)
# create a dataset loader
data_set = DatasetLoader(Data, current_task=0, transform=None, load_images=False, path=None)

# We can visualize samples from the sequence of tasks
for i in range(10):
    data_set.set_task(i)
    
    folder = "./Samples/disjoint_10_tasks/"
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    path_samples = os.path.join(folder, "MNIST_task_{}.png".format(i))
    data_set.visualize_sample(path_samples , number=100, shape=[28,28,1])
    
# use the dataset with pytorch dataloader for training an algo

# create pytorch dataloader
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

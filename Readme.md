## Continual Learning Data Former

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/9273eb0f97b946308248b0007e054e54)](https://app.codacy.com/app/TLESORT/Continual_Learning_Data_Former?utm_source=github.com&utm_medium=referral&utm_content=TLESORT/Continual_Learning_Data_Former&utm_campaign=Badge_Grade_Dashboard)
[![DOI](https://zenodo.org/badge/198824802.svg)](https://zenodo.org/badge/latestdoi/198824802)


### Intro

This repositery proprose several script to create sequence of tasks for continual learning. The spirit is the following : 
Instead of managing the sequence of tasks while learning, we create the sequence of tasks first and then we load tasks 
one by one while learning.

It makes programming easier and code cleaner.

### Installation

```bash
git clone https://github.com/TLESORT/Continual_Learning_Data_Former
cd Continual_Learning_Data_Former
git checkout Continuum
pip install .
```

### Use

```python
from continuum.disjoint import Disjoint
from torch.utils import data

# create continuum dataset
continuum = Disjoint(path=".", dataset="MNIST", task_number=10, download=True, train=True)

# create pytorch dataloader
train_loader = data.DataLoader(data_set, batch_size=64, shuffle=True, num_workers=6)

#set the task on 0 for example with the data_set
continuum.set_task(0)

# iterate on task 0
for t, (data, target) in enumerate(train_loader):
    print(target)
    
#change the task to 2 for example
continuum.set_task(2)

# iterate on task 2
for t, (data, target) in enumerate(train_loader):
    print(target)

# We can visualize samples from the sequence of tasks
for i in range(10):
    continuum.set_task(i)
    
    folder = "./Samples/disjoint_10_tasks/"
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    path_samples = os.path.join(folder, "MNIST_task_{}.png".format(i))
    continuum.visualize_sample(path_samples , number=100, shape=[28,28,1])
    
```


### Task sequences possibilities

-   **Disjoint tasks** : each task propose new classes
-   **Rotations tasks** : each tasks propose same data but with different rotations of datata point
-   **Permutations tasks** : each tasks propose same data but with different permutations of pixels
-   **Mnist Fellowship task** : each task is a new mnist like dataset (this sequence of task is an original contribution of this repository)

### An example with MNIST 5 dijoint tasks

|<img src="/Samples/disjoint_5_tasks/MNIST_task_0.png" width="150">|<img src="/Samples/disjoint_5_tasks/MNIST_task_1.png" width="150">|<img src="/Samples/disjoint_5_tasks/MNIST_task_2.png" width="150">|<img src="/Samples/disjoint_5_tasks/MNIST_task_3.png" width="150">|<img src="/Samples/disjoint_5_tasks/MNIST_task_4.png" width="150">|    
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
|Task 0 | Task 1 | Task 2 | Task 3 | Task 4|

More examples at [Samples](/Samples)

### Datasets

-   Mnist
-   fashion-Mnist
-   kmnist
-   cifar10
-   Core50/Core10

### Some supplementary option are possible
-   The number of tasks can be choosed (1, 3, 5 and 10 have been tested normally)
-   Classes order can be shuffled for disjoint tasks
-   We can choose the magnitude of rotation for rotations mnist

### Few possible invocations

-   Disjoint tasks

```python
#MNIST with 10 tasks of one class
continuum = Disjoint(path="./Data", dataset="MNIST", task_number=10, download=True, train=True)
```
-   Rotations tasks

```python
#MNIST with 5 tasks with various rotations
continuum = Rotations(path="./Data", dataset="MNIST", tasks_number=5, download=True, train=True, min_rot=0.0,
                 max_rot=90.0)
```

-   Permutations tasks

```python
#MNIST with 5 tasks with different permutations
continuum = Permutations(path="./Data", dataset="MNIST", tasks_number=1, download=False, train=True)
```



### Citing the Project

```Array.<string>
@software{timothee_lesort_2020_3605202,
  author       = {Timothée LESORT},
  title        = {Continual Learning Data Former},
  month        = jan,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {v1.0},
  doi          = {10.5281/zenodo.3605202},
  url          = {https://doi.org/10.5281/zenodo.3605202}
}

```



## Run Tests

Command to run from the main folder!

```bash
# First you need to generate the base datasets with the following commands
python main.py --dataset MNIST --n_tasks 10 --task disjoint
python main.py --dataset fashion --n_tasks 10 --task disjoint
python main.py --dataset kmnist --n_tasks 10 --task disjoint
python main.py --dataset MNIST --n_tasks 5 --task rotations
python main.py --dataset MNIST --n_tasks 5 --task permutations
python main.py --dataset MNIST --n_tasks 30 --task disjoint_rotations
python main.py --dataset MNIST  --n_tasks 10 --task disjoint_classes_permutations --index_permutations 2
python main.py --n_tasks 3 --task mnist_fellowship

python -m pytest tests/
python -m pytest --cov=. tests/
```

### Last Coverage

```buildoutcfg
----------- coverage: platform linux, python 3.6.8-final-0 -----------
Name                                                Stmts   Miss  Cover
-----------------------------------------------------------------------
Sequence_Formers/__init__.py                            0      0   100%
Sequence_Formers/disjoint.py                           14      2    86%
Sequence_Formers/disjoint_classes_permutations.py      32     32     0%
Sequence_Formers/disjoint_mnishion.py                  27     27     0%
Sequence_Formers/disjoint_rotations.py                 21      1    95%
Sequence_Formers/mnist_fellowship.py                   32     32     0%
Sequence_Formers/permutations.py                       17     17     0%
Sequence_Formers/rotations.py                          22      0   100%
Sequence_Formers/sequence_former.py                   103     11    89%
__init__.py                                            18      8    56%
data_loader.py                                        178     80    55%
data_utils.py                                         131     66    50%
dataset_loaders/LSUN.py                                19     16    16%
dataset_loaders/__init__.py                             0      0   100%
dataset_loaders/cifar10.py                             17      0   100%
dataset_loaders/core50.py                              44     37    16%
dataset_loaders/fashion.py                            110     73    34%
dataset_loaders/kmnist.py                               7      1    86%
main.py                                                58     58     0%
tests/__init__.py                                       0      0   100%
tests/test_Dataloader.py                               49      6    88%
tests/test_disjoint_former.py                          93     13    86%
tests/test_task_sequences.py                           68      5    93%
-----------------------------------------------------------------------
TOTAL                                                1060    485    54%

======= 105 passed, 4 skipped, 2 warnings in 100.94 seconds =======
```

### Build Doxygen doc

```bash
doxygen doxygen_config
```

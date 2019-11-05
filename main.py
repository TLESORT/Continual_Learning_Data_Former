import argparse
import os

from Sequence_Formers.disjoint import Disjoint
from Sequence_Formers.disjoint_rotations import DisjointRotations
from Sequence_Formers.disjoint_mnishion import DisjointMnishion
from Sequence_Formers.mnist_fellowship import MnistFellowship
from Sequence_Formers.rotations import Rotations
from Sequence_Formers.permutations import Permutations
from Sequence_Formers.disjoint_classes_permutations import DisjointClassesPermutations

from data_utils import check_args
import numpy as np
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--dir', default='./Archives', help='input directory')
parser.add_argument('--i', default='Data', help='input directory')

parser.add_argument('--index_permutation', default=None, type=int)
parser.add_argument('--task', default='disjoint',
                    choices=['rotations', 'permutations', 'disjoint_rotations', 'disjoint_mnishion',
                             'disjoint', "mnist_fellowship", 'disjoint_classes_permutations'],
                    help='type of task to create', )
parser.add_argument('--dataset', default='MNIST', type=str,
                    choices=['MNIST', 'fashion', 'core10', 'core50', 'cifar100', 'cifar10', 'mnishion', "kmnist", "mnist_fellowship"])
parser.add_argument('--n_tasks', default=3, type=int, help='number of tasks')
parser.add_argument('--num_classes', default=10, type=int, help='number of classes')
parser.add_argument('--classes_per_task', default=3, type=int, help='classes per tasks')
parser.add_argument('--imageSize', type=int, default=28, help='input batch size')
parser.add_argument('--img_channels', type=int, default=1, help='input batch size')
parser.add_argument('--min_rot', default=0., type=float, help='minimum rotation')
parser.add_argument('--max_rot', default=90., type=float, help='maximum rotation')
parser.add_argument('--number_rotation', default=1, type=int,
                    help='when we learn several disjoint sequence with different rotations')
parser.add_argument('--disjoint_classes', default=False, type=bool, help='In some scenari we may decide that classes'
                                                                         ' have same classes than previous task or '
                                                                         'new one (for example in Mnist fellowship)')
parser.add_argument('--path_only', action='store_true', default=False, help='when dataset is to big data may not be  '
                                                                            'preprocessed quickly, so we preprocess only '
                                                                            'images path and image will be loaded online')
args = parser.parse_args()

args = check_args(args)

seed = 0
torch.manual_seed(seed)
# parse arguments
torch.manual_seed(seed)
np.random.seed(seed)
args.gpu_mode = torch.cuda.is_available()

if args.gpu_mode:
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

print(str(args).replace(',', ',\n'))

args.i = os.path.join(args.dir, args.i)
args.o = os.path.join(args.i, 'Tasks', args.dataset)
args.i = os.path.join(args.i, 'Datasets')

if args.task == 'rotations':
    DataFormatter = Rotations(args)
elif args.task == 'permutations':
    DataFormatter = Permutations(args)
elif args.task == 'disjoint':
    DataFormatter = Disjoint(args)
elif args.task == 'mnist_fellowship':
    DataFormatter = MnistFellowship(args)
elif args.task == 'disjoint_rotations':
    DataFormatter = DisjointRotations(args)
elif args.task == 'disjoint_mnishion':
    DataFormatter = DisjointMnishion(args)
elif args.task == 'disjoint_classes_permutations':
    DataFormatter = DisjointClassesPermutations(args)
else:
    raise Exception("[!] There is no DataFormer  option for " + args.task)

DataFormatter.formating_data()

import torch
import os
from builders.continuumbuilder import ContinuumBuilder
from copy import deepcopy


class Permutations(ContinuumBuilder):
    '''Scenario : In this scenario, for each tasks all classes are available, however for each task pixels are permutated.
    The goal is to test algorithms where all data for each classes are not available simultaneously and are available from
     different mode of th distribution (different permutation modes).'''

    def __init__(self, path="./Data", dataset="MNIST", tasks_number=1, download=False, train=True):
        self.num_pixels = 0  # will be set in prepare_formatting
        self.perm_file = ""  # will be set in prepare_formatting
        self.list_perm = []

        super(Permutations, self).__init__(path=path,
                                           dataset=dataset,
                                           tasks_number=tasks_number,
                                           scenario="Rotations",
                                           download=download,
                                           train=train,
                                           num_classes=10)

    def prepare_formatting(self):

        self.num_pixels = self.imageSize * self.imageSize * self.img_channels
        self.perm_file = os.path.join(self.o, '{}_{}_train.pt'.format("ind_permutations", self.tasks_number))

        if os.path.isfile(self.perm_file):
            self.list_perm = torch.load(self.perm_file)
        else:
            p = torch.FloatTensor(range(self.num_pixels)).long()
            for _ in range(self.tasks_number):
                self.list_perm.append(p)
                p = torch.randperm(self.num_pixels).long().view(-1)
            torch.save(self.list_perm, self.perm_file)

    def transformation(self, ind_task, data):
        p = self.list_perm[ind_task]

        data = data.view(-1, self.num_pixels)
        return deepcopy(data).index_select(1, p).view(-1, self.img_channels, self.imageSize, self.imageSize)

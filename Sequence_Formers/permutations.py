import torch
from Sequence_Formers.sequence_former import Sequence_Former
from copy import deepcopy


class Permutations(Sequence_Former):

    def __init__(self, args):
        super(Permutations, self).__init__(args)

        self.list_perm = []

        self.num_pixels = self.imageSize * self.imageSize * self.img_channels

        p = torch.FloatTensor(range(self.num_pixels)).long()
        for _ in range(self.n_tasks):
            self.list_perm.append(p)
            p = torch.randperm(self.num_pixels).long().view(-1)

        torch.save(self.list_perm, self.o_train.replace('permutations', 'ind_permutations'))

    def transformation(self, ind_task, data):
        p = self.list_perm[ind_task]
        return deepcopy(data).index_select(1, p)

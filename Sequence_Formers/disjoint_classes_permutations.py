import os.path
import torch
from Sequence_Formers.disjoint import Disjoint


class DisjointClassesPermutations(Disjoint):
    def __init__(self, args):
        super(DisjointClassesPermutations, self).__init__(args)

        # There are several possible permutation inside the permutation file, so we need to choose one
        self.index_permutation = args.index_permutation
        if self.index_permutation is  None:
            raise AssertionError("index_permutation should be defined")
        self.permutation = torch.load("permutation_classes.t")[self.index_permutation]

        name = ''
        for i in range(10):
            name += str(int(self.permutation[i]))

        if self.path_only:
            light_id='_light'
        else:
            light_id=''

        self.o_train = os.path.join(self.o, '{}_{}_{}_train{}.pt'.format("disjoint", name, self.n_tasks, light_id))
        self.o_valid = os.path.join(self.o, '{}_{}_{}_valid{}.pt'.format("disjoint", name, self.n_tasks, light_id))
        self.o_test = os.path.join(self.o, '{}_{}_{}_test{}.pt'.format("disjoint", name, self.n_tasks, light_id))

        self.o_train_full = os.path.join(self.o, '{}_1-{}_{}_train{}.pt'.format("disjoint", name, self.n_tasks, light_id))
        self.o_valid_full = os.path.join(self.o, '{}_1-{}_{}_valid{}.pt'.format("disjoint", name, self.n_tasks, light_id))
        self.o_test_full = os.path.join(self.o, '{}_1-{}_{}_test{}.pt'.format("disjoint", name, self.n_tasks, light_id))


    def select_index(self, ind_task, y):

        if not self.n_tasks == self.num_classes:
            raise AssertionError("Other cases are not implemented yet")

        class_min = ind_task
        class_max = ind_task + 1
        wrong_class_min = int(self.permutation[ind_task])
        wrong_class_max = wrong_class_min + 1

        return class_min, class_max, ((y >= wrong_class_min) & (y < wrong_class_max)).nonzero().view(-1)

    def label_transformation(self, ind_task, label):
        """
        Apply transformation to label if needed
        :param ind_task: task index in the sequence
        :param label: label to process
        :return: data post processing
        """

        wrong_label = label.clone().fill_(ind_task)
        return wrong_label

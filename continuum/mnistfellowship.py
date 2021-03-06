import torch
import os

from .data_utils import load_data
from .continuumbuilder import ContinuumBuilder



class MnistFellowship(ContinuumBuilder):
    def __init__(self, path="./Archives/Data", tasks_number=3, merge=False, download=False, train=True):

        self.merge = merge
        if self.merge:
            self.scenario = "mnist_fellowship_merge"
        else:
            self.scenario = "mnist_fellowship"

        super(MnistFellowship, self).__init__(path=path,
                                              dataset="mnist_fellowship",
                                              tasks_number=tasks_number,
                                              scenario=self.scenario,
                                              download=download,
                                              train=train,
                                              num_classes=10)

    def select_index(self, ind_task, y):

        if not self.merge:
            class_min = self.num_classes * ind_task
            class_max = self.num_classes * (ind_task + 1) - 1
        else:
            class_min = 0
            class_max = self.num_classes - 1
        return class_min, class_max, torch.arange(len(y))

    def label_transformation(self, ind_task, label):
        """
        Apply transformation to label if needed
        :param ind_task: task index in the sequence
        :param label: label to process
        :return: data post processing
        """

        # if self.disjoint class 0 of second task become class 10, class 1 -> class 11, ...
        if not self.merge:
            label = label + self.num_classes * ind_task

        return label

    def create_task(self, ind_task, x_, y_):

        if ind_task == 0:  # MNIST
            self.dataset = 'MNIST'
        elif ind_task == 1:  # fashion
            self.dataset = 'fashion'
        elif ind_task == 2:  # kmnist
            self.dataset = 'kmnist'

        # we load a new dataset for each task
        x_, y_ = load_data(self.dataset, self.i)

        return super().create_task(ind_task, x_, y_)

import torch

if os.path.exists("Sequence_Formers"):
    from data_utils import load_data
    from Sequence_Formers.sequence_former import Sequence_Former
else:
    from ..data_utils import load_data


class MnistFellowship(Sequence_Former):
    def __init__(self, args):
        super(MnistFellowship, self).__init__(args)

        self.disjoint_classes = args.disjoint_classes

        if not self.num_classes == 10:
            raise AssertionError("Wrong number of classes for this experiment")

    def select_index(self, ind_task, y):

        if self.disjoint_classes:
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
        if self.disjoint_classes:
            label = label + self.num_classes * ind_task

        return label

    def create_task(self, ind_task, x_tr, y_tr, x_te, y_te):

        if ind_task == 0:  # MNIST
            self.dataset = 'MNIST'
        elif ind_task == 1:  # fashion
            self.dataset = 'fashion'
        elif ind_task == 2:  # kmnist
            self.dataset = 'kmnist'

        # we load a new dataset for each task
        x_tr, y_tr, x_te, y_te = load_data(self.dataset, self.i, self.imageSize, self.path_only)

        return super().create_task(ind_task, x_tr, y_tr, x_te, y_te)

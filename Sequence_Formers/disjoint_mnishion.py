
if os.path.exists("Sequence_Formers"):
    from data_utils import load_data
    from Sequence_Formers.sequence_former import Sequence_Former
else:
    from ..data_utils import load_data



class DisjointMnishion(Sequence_Former):
    """Scenario : in this scenario we learn classes one by one first with MNIST and then with fashion-MNIST"""

    def __init__(self, args):
        super(DisjointMnishion, self).__init__(args)

        if not self.n_tasks == 20:
            raise AssertionError("Wrong number of tasks for this experiment")

    def select_index(self, ind_task, y):

        if not self.num_classes == 10:
            raise AssertionError("Wrong number of classes for this experiment")

        class_min = ind_task % self.num_classes
        class_max = class_min + 1

        return class_min, class_max, ((y >= class_min) & (y < class_max)).nonzero().view(-1)

    def label_transformation(self, ind_task, label):
        """
        Apply transformation to label if needed
        :param ind_task: task index in the sequence
        :param label: label to process
        :return: data post processing
        """

        # if self.disjoint class 0 of second task become class 10, class 1 -> class 11, ...
        if self.disjoint_classes:
            label = label.clone().fill_(ind_task)
        return label

    def create_task(self, ind_task, x_tr, y_tr, x_te, y_te):

        if ind_task < 10:  # MNIST
            self.dataset = 'MNIST'
        elif ind_task >= 10:  # fashion
            self.dataset = 'fashion'

        # we reload a new dataset for each sequence of 10 tasks
        x_tr, y_tr, x_te, y_te = load_data(self.dataset, self.i, self.imageSize, self.path_only)

        return super().create_task(ind_task, x_tr, y_tr, x_te, y_te)

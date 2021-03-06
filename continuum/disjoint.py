from continuum.continuumbuilder import ContinuumBuilder


class Disjoint(ContinuumBuilder):
    """Scenario : each new classes gives never seen new classes to learn. The code here allows to choose in how many task we
     want to split a dataset and therefor in autorize to choose the number of classes per tasks.
    This scenario test algorithms when there is no intersection between tasks."""

    def __init__(self, path="./Data", dataset="MNIST", tasks_number=1, download=False, train=True):
        super(Disjoint, self).__init__(path=path,
                                       dataset=dataset,
                                       tasks_number=tasks_number,
                                       scenario="Disjoint",
                                       download=download,
                                       train=train,
                                       num_classes=10)

    def select_index(self, ind_task, y):
        cpt = int(self.num_classes / self.tasks_number)

        if not cpt > 0:
            raise AssertionError("Cpt can't be equal to zero for selection of classes")

        class_min = ind_task * cpt
        class_max = (ind_task + 1) * cpt

        return class_min, class_max, ((y >= class_min) & (y < class_max)).nonzero().view(-1)

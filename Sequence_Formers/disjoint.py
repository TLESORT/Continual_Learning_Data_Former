
import os

if os.path.exists("Sequence_Formers"):
    from Sequence_Formers.sequence_former import Sequence_Former
else:
    from ..Sequence_Formers.sequence_former import Sequence_Former


"""
Scenario : each new classes gives never seen new classes to learn. The code here allows to choose in how many task we
 want to split a dataset and therefor in autorize to choose the number of classes per tasks.
This scenario test algorithms when there is no intersection between tasks.
"""

class Disjoint(Sequence_Former):

    def __init__(self, args):
        super(Disjoint, self).__init__(args)

    def select_index(self, ind_task, y):
        cpt = int(self.num_classes / self.n_tasks)

        if not cpt > 0:
            raise AssertionError("Cpt can't be equal to zero for selection of classes")

        class_min = ind_task * cpt
        class_max = (ind_task + 1) * cpt

        return class_min, class_max, ((y >= class_min) & (y < class_max)).nonzero().view(-1)


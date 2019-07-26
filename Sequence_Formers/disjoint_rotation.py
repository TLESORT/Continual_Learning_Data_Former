from Sequence_Formers.rotations import Rotations

"""
Scenario : we have several disjoint sequence but each sequence is rotated a bit  
This scenario test algorithms when there are both partially disjoint tasks and concept drift.
"""


class Disjoint_rotation(Rotations):
    def __init__(self, args):
        super(Disjoint_rotation, self).__init__(args)
        self.number_rotation = 3

        assert self.num_classes * self.number_rotation == self.n_tasks

    def transformation(self, ind_task, data):
        """
        This function apply a different rotation for each disjoint sequence
        :param ind_task: gives the task index
        :param data:  give data to modify
        :return:
        """

        ind_rotation = ind_task // self.num_classes

        delta_rot = 1.0 * (self.max_rot - self.min_rot) / self.number_rotation

        noise = 1.0 * delta_rot / 10.0

        min_rot = self.min_rot + (delta_rot * ind_rotation) - noise
        max_rot = self.min_rot + (delta_rot * ind_rotation) + noise

        if ind_rotation != 0:
            data = self.apply_rotation(data, min_rot, max_rot)

        return data

    def select_index(self, ind_task, y):
        # remap index between 0 and self.num_classes -1
        ind_selection = ind_task % self.num_classes

        class_min = ind_selection
        class_max = ind_selection + 1

        return class_min, class_max, ((y >= class_min) & (y < class_max)).nonzero().view(-1)

from torchvision import transforms
import torch
from Sequence_Formers.sequence_former import Sequence_Former

"""
Scenario : In this scenario, for each tasks all classes are available, however for each task data rotate a bit.
The goal is to test algorithms where all data for each classes are not available simultaneously and there is a concept drift.
"""



class Rotations(Sequence_Former):
    def __init__(self, args):
        super(Rotations, self).__init__(args)

        self.max_rot = args.max_rot
        self.min_rot = args.min_rot

    def apply_rotation(self, data, min_rot, max_rot):

        transform = transforms.Compose(
            [transforms.RandomAffine(degrees=[min_rot, max_rot]),
             transforms.ToTensor()])

        result = torch.FloatTensor(data.size(0), 784)
        for i in range(data.size(0)):
            X = data[i].view(self.imageSize, self.imageSize)
            X = transforms.ToPILImage()(X)
            result[i] = transform(X).view(784)

        return result

    def transformation(self, ind_task, data):

        delta_rot = 1.0 * (self.max_rot - self.min_rot) / self.n_tasks
        noise = 1.0 * delta_rot / 10.0

        min_rot = self.min_rot + (delta_rot * ind_task) - noise
        max_rot = self.min_rot + (delta_rot * ind_task) + noise

        return self.apply_rotation(data, min_rot, max_rot)

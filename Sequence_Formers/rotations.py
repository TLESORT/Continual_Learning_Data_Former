from torchvision import transforms
import torch
from Sequence_Formers.sequence_former import Sequence_Former


class Rotations(Sequence_Former):
    '''Scenario : In this scenario, for each tasks all classes are available, however for each task data rotate a bit.
    The goal is to test algorithms where all data for each classes are not available simultaneously and there is a concept
     drift.'''

    def __init__(self, path="./Data", dataset="MNIST", tasks_number=1, download=False, train=True, min_rot=0.0,
                 max_rot=90.0):
        self.max_rot = max_rot
        self.min_rot = min_rot

        super(Rotations, self).__init__(path=path,
                                        dataset=dataset,
                                        tasks_number=tasks_number,
                                        scenario="Rotations",
                                        download=download,
                                        train=train,
                                        num_classes=10)

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
        delta_rot = 1.0 * (self.max_rot - self.min_rot) / self.tasks_number
        noise = 1.0 * delta_rot / 10.0

        min_rot = self.min_rot + (delta_rot * ind_task) - noise
        max_rot = self.min_rot + (delta_rot * ind_task) + noise

        return self.apply_rotation(data, min_rot, max_rot)



import os

def check_task_sequences_files(scenario, folder,n_tasks, dataset, train=True):

    if train:
        filename = "{}_{}_{}.pt".format(scenario, n_tasks, "train")
    else:
        filename = "{}_{}_{}.pt".format(scenario, n_tasks, "test")

    path = os.path.join(folder, "Data", "Continua", dataset, filename)
    if not os.path.isfile(path):
        raise AssertionError("Test fail with file : {}".format(path))



class args(object):
    def __init__(self):
        self.n_tasks = 10
        self.num_classes = 10
        self.task = "disjoint"
        self.img_channels = 1
        self.imageSize = 28
        self.dir = None
        self.dataset = None
        self.path_only = None
        self.verbose = False

    def set_paths(self):

        if self.dir is None:
            raise AssertionError("Dir is None")
        if self.dataset is None:
            raise AssertionError("dataset is None")
        self.o = os.path.join(self.dir, 'Data', 'Continua', self.dataset)
        self.i = os.path.join(self.dir, 'Data', 'Datasets')

    def add_supp_parameters(self, method):

        if "rotation" in method:
            self.min_rot = 0.0
            self.max_rot = 90.0
import os.path
import torch
from copy import deepcopy

if os.path.exists("Sequence_Formers"):  # check if we are in the folder Continual_Learning_Data_Former
    from data_utils import load_data, check_and_Download_data
else:
    from ..data_utils import load_data, check_and_Download_data


class Sequence_Former(object):
    '''Parent Class for Sequence Formers'''

    def __init__(self, args):
        super(Sequence_Former, self).__init__()

        self.n_tasks = args.n_tasks
        self.num_classes = args.num_classes
        self.i = args.i
        self.o = args.o
        self.imageSize = args.imageSize
        self.img_channels = args.img_channels
        self.dataset = args.dataset
        self.path_only = args.path_only  # only valid for core50 at the moment
        self.task = args.task
        self.verbose = args.verbose

        # if self.path_only we don't load data but just path
        # data will be loaded online while learning
        # it is considered as light mode this continual dataset are easy to generate and load
        if self.path_only:
            light_id = '_light'
        else:
            light_id = ''

        if not os.path.exists(args.o):
            os.makedirs(args.o)

        self.o_train = os.path.join(self.o, '{}_{}_train{}.pt'.format(self.task, self.n_tasks, light_id))
        self.o_valid = os.path.join(self.o, '{}_{}_valid{}.pt'.format(self.task, self.n_tasks, light_id))
        self.o_test = os.path.join(self.o, '{}_{}_test{}.pt'.format(self.task, self.n_tasks, light_id))

        self.o_train_full = os.path.join(self.o, '{}_1-{}_train{}.pt'.format(self.task, self.n_tasks, light_id))
        self.o_valid_full = os.path.join(self.o, '{}_1-{}_valid{}.pt'.format(self.task, self.n_tasks, light_id))
        self.o_test_full = os.path.join(self.o, '{}_1-{}_test{}.pt'.format(self.task, self.n_tasks, light_id))

        check_and_Download_data(self.i, self.dataset, task=self.task)

    def select_index(self, ind_task, y):
        """
        This function help to select data in particular if needed
        :param ind_task: task index in the sequence
        :param y: data label
        :return: class min, class max, and index of data to keep
        """
        return 0, self.num_classes - 1, torch.arange(len(y))

    def transformation(self, ind_task, data):
        """
        Apply transformation to data if needed
        :param ind_task: task index in the sequence
        :param data: data to process
        :return: data post processing
        """
        if not ind_task < self.n_tasks:
            raise AssertionError("Error in task indice")
        return deepcopy(data)

    def label_transformation(self, ind_task, label):
        """
        Apply transformation to label if needed
        :param ind_task: task index in the sequence
        :param label: label to process
        :return: data post processing
        """
        if not ind_task < self.n_tasks:
            raise AssertionError("Error in task indice")
        return label

    @staticmethod
    def get_valid_ind(i_tr):
        # it is time to taxe train for validation
        len_valid = int(len(i_tr) * 0.2)
        indices = torch.randperm(len(i_tr))

        valid_ind = indices[:len_valid]
        train_ind = indices[len_valid:]

        i_va = i_tr[valid_ind]
        i_tr = i_tr[train_ind]

        return i_tr, i_va

    def create_task(self, ind_task, x_tr, y_tr, x_te, y_te):

        # select only the good classes
        class_min, class_max, i_tr = self.select_index(ind_task, y_tr)
        _, _, i_te = self.select_index(ind_task, y_te)

        i_tr, i_va = self.get_valid_ind(i_tr)

        x_tr_t = self.transformation(ind_task, x_tr[i_tr])
        x_va_t = self.transformation(ind_task, x_tr[i_va])
        x_te_t = self.transformation(ind_task, x_te[i_te])

        y_tr_t = self.label_transformation(ind_task, y_tr[i_tr])
        y_va_t = self.label_transformation(ind_task, y_tr[i_va])
        y_te_t = self.label_transformation(ind_task, y_te[i_te])

        if self.verbose and self.path_only:
            print("Task : {}".format(ind_task))
            ind = torch.randperm(len(x_tr_t))[:10]
            print(x_tr_t[ind])
            ind = torch.randperm(len(x_va_t))[:10]
            print(x_va_t[ind])
            ind = torch.randperm(len(x_te_t))[:10]
            print(x_te_t[ind])

        return class_min, class_max, x_tr_t, y_tr_t, x_va_t, y_va_t, x_te_t, y_te_t

    def formating_data(self):

        # variable to save the sequence
        tasks_tr = []
        tasks_va = []
        tasks_te = []

        # variable to save the cumul of the sequence for upperbound
        tasks_tr_full = []
        tasks_va_full = []
        tasks_te_full = []
        full_x_tr, full_y_tr = None, None
        full_x_va, full_y_va = None, None
        full_x_te, full_y_te = None, None

        x_tr, y_tr, x_te, y_te = load_data(self.dataset, self.i)

        for ind_task in range(self.n_tasks):

            c1, c2, x_tr_t, y_tr_t, x_va_t, y_va_t, x_te_t, y_te_t = self.create_task(ind_task, x_tr, y_tr, x_te, y_te)

            tasks_tr.append([(c1, c2), x_tr_t, y_tr_t])
            tasks_va.append([(c1, c2), x_va_t, y_va_t])
            tasks_te.append([(c1, c2), x_te_t, y_te_t])

            if ind_task == 0:
                full_x_tr = x_tr_t
                full_x_va = x_va_t
                full_x_te = x_te_t
                full_y_tr = y_tr_t
                full_y_va = y_va_t
                full_y_te = y_te_t

        if not self.path_only:
            print(tasks_tr[0][1].shape)
            print(tasks_tr[0][1].mean())
            print(tasks_tr[0][1].std())

        torch.save(tasks_tr, self.o_train)
        torch.save(tasks_va, self.o_valid)
        torch.save(tasks_te, self.o_test)

        tasks_tr_full.append([(0, self.num_classes), full_x_tr, full_y_tr])
        tasks_va_full.append([(0, self.num_classes), full_x_va, full_y_va])
        tasks_te_full.append([(0, self.num_classes), full_x_te, full_y_te])

        torch.save(tasks_tr_full, self.o_train_full)
        torch.save(tasks_va_full, self.o_valid_full)
        torch.save(tasks_te_full, self.o_test_full)

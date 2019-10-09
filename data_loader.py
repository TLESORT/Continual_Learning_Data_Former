import torch
import numpy as np
from copy import deepcopy
from torch.utils import data
import torchvision.transforms.functional as TF
from PIL import Image
import os

if os.path.exists("Sequence_Formers"):
    from data_utils import make_samples_batche, save_images
else:
    from .data_utils import make_samples_batche, save_images


class DatasetLoader(data.Dataset):
    def __init__(self, data, current_task=0, transform=None, load_images=False, path=None):

        '''

        dataset.shape = [num , 3, image_number]
        dataset[0 , 1, :] # all data from task 0
        dataset[0 , 2, :] # all label from task 0

        '''

        self.dataset = data
        self.n_tasks = len(self.dataset)

        self.current_task = current_task
        self.transform = transform
        self.load_images = load_images
        self.path = path

        'Initialization'
        if self.load_images:
            list_data = range(len(self.dataset[self.current_task][1]))
        else:
            list_data = range(self.dataset[self.current_task][1].shape[0])
        list_labels = self.dataset[self.current_task][2].tolist()

        # convert it to dictionnary for pytorch
        self.list_IDs = {i: list_data[i] for i in range(0, len(list_data))}
        self.labels = {i: list_labels[i] for i in range(0, len(list_labels))}

    def __len__(self):
        return len(self.dataset[self.current_task][1])

    def __getitem__(self, index):
        'Generates one sample of data'

        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label

        if self.load_images:
            X = Image.open(os.path.join(self.path, self.dataset[self.current_task][1][ID])).convert('RGB')
        else:
            # here they have been already loaded, so I don't know if it is really optimized....
            X = self.dataset[self.current_task][1][ID]
        y = self.labels[ID]

        if self.transform is not None:
            if not self.load_images:
                X = TF.to_pil_image(X).convert('RGB')
            X = self.transform(X)

        return X, y

    def next(self):
        return self.__next__()

    def set_task(self, new_task_index):
        """

        :param new_task_index:
        :return:
        """
        self.current_task = new_task_index

        list_data = range(self.dataset[self.current_task][1].shape[0])
        list_labels = self.dataset[self.current_task][2].tolist()

        # convert it to dictionnary for pytorch
        self.list_IDs = {i: list_data[i] for i in range(0, len(list_data))}
        self.labels = {i: list_labels[i] for i in range(0, len(list_labels))}

        return self

    def shuffle_task(self):
        indices = torch.randperm(len(self.dataset[self.current_task][1]))
        self.dataset[self.current_task][1] = self.dataset[self.current_task][1][indices].clone()
        self.dataset[self.current_task][2] = self.dataset[self.current_task][2][indices].clone()

        return self

    def get_sample(self, number, shape):
        """
        This function return a number of sample from the dataset
        :param number: number of data point expected
        :return: FloatTensor on cpu of all samples
        """

        indices = torch.randperm(len(self.labels))[0:number]
        batch = None

        for i in range(number):
            # we need to use get item to have the transform used
            img, y = self.__getitem__(indices[i].item())

            if i == 0:
                batch = torch.FloatTensor(number, shape[0], shape[1], shape[2])

            batch[i] = img.reshape(shape).clone()

        labels = self.dataset[self.current_task][2][indices]
        return batch, labels

    def get_batch_from_label(self, label):
        """
        This function return a number of sample from the dataset with specific label
        :param label: label to get data from
        :return: FloatTensor on cpu of all samples
        """
        indices = [i for i, y in enumerate(self.dataset[self.current_task][2]) if y == label]

        batch = None

        for i in range(len(indices)):
            # we need to use get item to have the transform used
            img, y = self.__getitem__(indices[i])

            if i == 0:
                batch = torch.FloatTensor(len(indices), img.shape[0])

            batch[i] = img.clone()

        labels = self.dataset[self.current_task][2][indices]
        return batch, labels

    def concatenate(self, new_data, task=0):

        '''

        :param new_data: data to add to the actual task
        :return: the actual dataset with supplementary data inside
        '''
        self.dataset[self.current_task][1] = torch.cat((self.dataset[self.current_task][1], new_data.dataset[task][1]),
                                                       0).clone()
        self.dataset[self.current_task][2] = torch.cat((self.dataset[self.current_task][2], new_data.dataset[task][2]),
                                                       0).clone()

        return self

    def free(self, ind_task):

        self.dataset[ind_task][1] = None
        self.dataset[ind_task][2] = None

    def get_current_task(self):
        return self.current_task

    def save(self, path, force=False):
        if force:
            torch.save(self.dataset, path)
        else:
            print("WE DO NOT SAVE ANYMORE")

    def visualize_sample(self, path, number, shape, class_=None):

        data, target = self.get_sample(number, shape)

        # get sample in order from 0 to 9
        target, order = target.sort()
        data = data[order]
        #
        # if self.transform is not None:
        #     tf_bacth = None
        #     for i in range(number):
        #          tf_data = TF.to_pil_image(data[i])
        #          tf_data = self.transform(tf_data)
        #          if i == 0:
        #             tf_bacth = torch.FloatTensor(number, tf_data.shape[0], tf_data.shape[1], tf_data.shape[2])
        #          tf_bacth[i] = tf_data
        #     data = tf_bacth

        image_frame_dim = int(np.floor(np.sqrt(number)))

        if shape[2] == 1:
            data_np = data.numpy().reshape(number, shape[0], shape[1], shape[2])
            save_images(data_np[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                        path)
        elif shape[2] == 3:
            # data = data.numpy().reshape(number, shape[0], shape[1], shape[2])
            # if self.dataset_name == 'cifar10':
            data = data.numpy().reshape(number, shape[2], shape[1], shape[0])
            # data = data.numpy().reshape(number, shape[0], shape[1], shape[2])

            # remap between 0 and 1
            # data = data - data.min()
            # data = data / data.max()

            data = data / 2 + 0.5  # unnormalize
            make_samples_batche(data[:number], number, path)
        else:
            save_images(data[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                        path)

        return data

    def visualize_reordered(self, path, number, shape, permutations):

        data = self.visualize_sample(path, number, shape)

        data = data.reshape(-1, shape[0] * shape[1] * shape[2])
        concat = deepcopy(data)

        image_frame_dim = int(np.floor(np.sqrt(number)))

        for i in range(1, self.n_tasks):
            _, inverse_permutation = permutations[i].sort()
            reordered_data = deepcopy(data.index_select(1, inverse_permutation))
            print("hope you're in shape")
            print(reordered_data.shape)
            print(concat.shape)
            concat = torch.cat((concat, reordered_data), 0)

        if shape[2] == 1:
            concat = concat.numpy().reshape(number * self.n_tasks, shape[0], shape[1], shape[2])
            save_images(concat[:image_frame_dim * image_frame_dim * self.n_tasks, :, :, :],
                        [self.n_tasks * image_frame_dim, image_frame_dim],
                        path)
        else:
            concat = concat.numpy().reshape(number * self.n_tasks, shape[2], shape[1], shape[0])
            make_samples_batche(concat[:self.batch_size], self.batch_size, path)

    def increase_size(self, increase_factor):
        # first data
        self.dataset[self.current_task][1] = torch.cat([self.dataset[self.current_task][1]] * increase_factor, 0)
        # then labels
        self.dataset[self.current_task][2] = torch.cat([self.dataset[self.current_task][2]] * increase_factor, 0)

        return self

    def sub_sample(self, number):
        indices = torch.randperm(len(self))[0:number]
        self.dataset[self.current_task][1] = self.dataset[self.current_task][1][indices]
        self.dataset[self.current_task][2] = self.dataset[self.current_task][2][indices]
        return self

    def delete_class(self, class_ind):
        # select all the classes differnet to ind_class
        indices = torch.nonzero(self.dataset[self.current_task][2] != class_ind)
        indices = indices.reshape(-1)

        # keep only those indices
        self.dataset[self.current_task][1] = self.dataset[self.current_task][1][indices]
        self.dataset[self.current_task][2] = self.dataset[self.current_task][2][indices]

    def delete_task(self, ind_task):
        # keep only those indices
        self.dataset[ind_task][1] = []
        self.dataset[ind_task][2] = []

import os.path
import torch
import numpy as np
import pickle as pkl


def create_data_sets(path, num_classes):
    """
    This function create test and train sets for core50
    data and paths.pkl need to be downladed manually at "https://vlomonaco.github.io/core50/index.html#dataset"
    :param path: path to the folder with all data and paths.pkl
    :return: None
    """

    name_dataset = "core" + str(num_classes)

    if not (num_classes == 10 or num_classes == 50):
        raise AssertionError("Only 10 or 50 are possible here")

    image_path = path.replace("core10", "core50")
    path_path = os.path.join(path, 'paths.pkl').replace("core10", "core50")

    if not os.path.isfile(path_path):
        raise AssertionError("paths.pkl have to be downloaded in https://vlomonaco.github.io/core50/index.html#dataset"
                             " and put in {}".format(path_path))

    # test if all folders exists
    folders_exists = True
    # 11 sequences
    for i in range(1, 11):
        # 50 objects
        for j in range(1, 50):
            folder = os.path.join(image_path, "s"+str(i), "o"+str(j))
            if not os.path.isdir(folder):
                print("Missing folder {}".format(folder))
                folders_exists = False
    if not folders_exists:
        raise AssertionError("Some folder are missing and probable some data to download then in"
                             " https://vlomonaco.github.io/core50/index.html#dataset"
                             " and put in{}".format(image_path))


    pkl_file = open(path_path, 'rb')
    paths = pkl.load(pkl_file)

    ###################  Reduction of data size
    new_path = []
    for i in range(len(paths)):
        # we go from 20 Hz to 5 hz following https://arxiv.org/pdf/1805.10966.pdf
        if i % 4 == 0:
            new_path.append(paths[i])
    paths = new_path

    # ex : paths[0] -> 's11/o1/C_11_01_000.png'

    # [o1, ..., o5] -> plug adapters  -> label 1
    # [o6, ..., o10] -> mobile phones
    # [o11, ..., o15] -> scissors
    # [o16, ..., o20] -> light bulbs
    # [o21, ..., o25] -> cans
    # [o26, ..., o30] -> glasses
    # [o31, ..., o35] -> balls
    # [o36, ..., o40] -> markers
    # [o41, ..., o45] -> cups
    # [o46, ..., o50] -> remote controls

    # first : get labels
    list_label = []
    for i in range(len(paths)):
        str_path = paths[i]  # Ex: 's11/o1/C_11_01_000.png'
        str_label = str_path.split('/')[1]  # -> 'o1'
        int_label = int(str_label.replace('o', ''))  # -> 1

        # We remap from 1 to 50 from 0 to 9
        if num_classes == 10:
            list_label.append((int_label - 1) // 5)
        else:  # We remap from 1 to 50 from 0 to 49
            list_label.append(int_label - 1)

    # second : separate test (sequences #3, #7, #10) from train
    list_train = []
    list_test = []

    for i in range(len(paths)):
        str_path = paths[i]
        str_sequence = str_path.split('/')[0]
        int_sequence = int(str_sequence.replace('s', ''))

        # sequence 3,7,10 are for test as in the original paper
        if int_sequence == 3 or int_sequence == 7 or int_sequence == 10:
            list_test.append(i)
        elif int_sequence <= 11:
            list_train.append(i)
        else:
            print("There is a problem")

    train_label = np.zeros(len(list_train))
    train_path = []

    test_label = np.zeros(len(list_test))
    test_path = []

    print("We start creating the train set")

    # train data
    for i in range(len(list_train)):
        ind = list_train[i]
        label = list_label[ind]
        train_label[i] = label
        train_path.append(os.path.join(image_path, paths[ind]))

    save_path = path.replace("raw", "processed")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.savez(os.path.join(save_path, name_dataset+'_paths_train.npz'), y=train_label, paths=train_path)

    print("We start creating the test set")

    # test data
    for i in range(len(list_test)):
        ind = list_test[i]
        label = list_label[ind]
        test_label[i] = label
        test_path.append(os.path.join(image_path, paths[ind]))

    np.savez(os.path.join(save_path, name_dataset+'_paths_test.npz'), y=test_label, paths=test_path)


def load_path(path):
    path_tr = np.load(path)['paths']
    y_tr = np.load(path)['y']
    y_tr = y_tr.reshape((-1))
    y_tr = torch.Tensor(y_tr)
    return path_tr, y_tr


def load_core50(dataset, path):
    """
    Function to load data from core50. Actually we only process path to data and not data for efficiency purpose.
    :param dataset: allow to know if we are loading core10 or cor50
    :param path: path to data
    :param path_only:
    :return:
    """

    path_raw = os.path.join(path, dataset, "raw")
    path = os.path.join(path, dataset, "processed")

    path_train = os.path.join(path, '{}_paths_train.npz'.format(dataset))
    path_test = os.path.join(path, '{}_paths_test.npz'.format(dataset))

    if not (os.path.isfile(path_train) and os.path.isfile(path_test)):
        pass

    if dataset == "core50":
        create_data_sets(path_raw, 50)
    elif dataset == "core10":
        create_data_sets(path_raw, 10)
    else:
        raise AssertionError("Only core10 or core50 are possible here")

    x_tr, y_tr = load_path(path_train)
    x_te, y_te = load_path(path_test)

    return x_tr, y_tr, x_te, y_te

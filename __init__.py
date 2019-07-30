

try:
    from Continual_Learning_Data_Former.Sequence_Formers.sequence_former import Sequence_Former
    from Continual_Learning_Data_Former.Sequence_Formers.disjoint import Disjoint
    from Continual_Learning_Data_Former.dataset_loaders.kmnist import Kmnist
    from Continual_Learning_Data_Former.dataset_loaders.fashion import Fashion
    from Continual_Learning_Data_Former.dataset_loaders.LSUN import load_LSUN
    from Continual_Learning_Data_Former.dataset_loaders.cifar10 import load_Cifar10
    from Continual_Learning_Data_Former.dataset_loaders.core50 import load_core50
    from Continual_Learning_Data_Former.data_utils import make_samples_batche, save_images, load_data, check_and_Download_data
except:
    from Sequence_Formers.sequence_former import Sequence_Former
    from Sequence_Formers.disjoint import Disjoint
    from dataset_loaders.kmnist import Kmnist
    from dataset_loaders.fashion import Fashion
    from dataset_loaders.LSUN import load_LSUN
    from dataset_loaders.cifar10 import load_Cifar10
    from dataset_loaders.core50 import load_core50
    from data_utils import make_samples_batche, save_images, load_data, check_and_Download_data

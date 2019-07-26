

try:
    from Continual_Learning_Data_Former.Sequence_Formers.disjoint import Disjoint
    from Continual_Learning_Data_Former.dataset_loaders.LSUN import load_LSUN
    from Continual_Learning_Data_Former.dataset_loaders.cifar10 import load_Cifar10
    from Continual_Learning_Data_Former.dataset_loaders.core50 import load_core50
    from Continual_Learning_Data_Former.data_utils import make_samples_batche, save_images
except:
    from Sequence_Formers.disjoint import Disjoint
    from dataset_loaders.LSUN import load_LSUN
    from dataset_loaders.cifar10 import load_Cifar10
    from dataset_loaders.core50 import load_core50
    from data_utils import make_samples_batche, save_images

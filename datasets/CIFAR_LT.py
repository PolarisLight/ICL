import os
import shutil

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
import  matplotlib.pyplot as plt
from PIL import Image
from utils.autoaugment import CIFAR10Policy, Cutout

def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

# data transform settings
normalize = transforms.Normalize(
    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
data_transforms = {
    'base_train': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]),
    # augmentation adopted in balanced meta softmax & NCL
    'advanced_train': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        normalize
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
}

class CIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10
    def __init__(self, args,train=True):
        root = args.dataset.data_dir
        transforms_type = 'advanced_train' if train else 'test'
        transform = data_transforms[transforms_type]
        target_transform = None
        download = True

        super(CIFAR10, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index):
        data = super().__getitem__(index)
        return {'image': data[0], 'label': data[1]}

class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100


class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10
    def __init__(self, args,train=True):
        root = args.dataset.data_dir
        imb_factor = args.dataset.imb_factor
        imb_type = 'exp'
        transforms_type = 'advanced_train' if train else 'test'
        transform = data_transforms[transforms_type]
        target_transform = None
        download = True

        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        # print(rand_number)
        np.random.seed(0)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        if train:
            self.gen_imbalanced_data(img_num_list)


    def gen_balanced_data(self):
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        
        self.num_per_cls_dict = {cls: sum(targets_np == cls) for cls in classes}
        print(f"Balanced dataset size: {len(self.data)}")
    

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            # np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        print(len(self.data))
        self.targets = to_categorical(new_targets)

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list
    
    def __getitem__(self, index):
        data = super().__getitem__(index)
        return {'image': data[0], 'label': data[1]}

    def get_data_by_class(self, class_idx, num_samples):
        """
        Get a specified number of samples from a given class, with transform applied.

        Args:
            class_idx (int): The class index to fetch data from.
            num_samples (int): Number of samples to fetch.

        Returns:
            dict: {'image': transformed images, 'label': corresponding labels}.
        """
        if class_idx not in self.num_per_cls_dict:
            raise ValueError(f"Class index {class_idx} is not valid. Choose from 0 to {self.cls_num - 1}.")
        
        # Step 1: Fetch indices for the specified class
        indices = np.where(np.array(self.targets) == class_idx)[0]
        if len(indices) < num_samples:
            raise ValueError(f"Requested {num_samples} samples, but only {len(indices)} are available in class {class_idx}.")

        # Step 2: Select the required number of samples
        selected_indices = indices[:num_samples]
        data = self.data[selected_indices]
        targets = np.array(self.targets)[selected_indices]

        # Step 3: Apply transformations
        transformed_images = []
        for img in data:
            # Convert to PIL image for transformation compatibility
            pil_image = Image.fromarray(img)
            if self.transform is not None:
                transformed_images.append(self.transform(pil_image))
            else:
                transformed_images.append(pil_image)

        # Convert to tensor if necessary (transform typically handles this)
        transformed_images = torch.stack(transformed_images) if isinstance(transformed_images[0], torch.Tensor) else transformed_images

        return {'image': transformed_images, 'label': torch.tensor(targets, dtype=torch.long)}



class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100

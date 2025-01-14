import os
import math
import shutil

import numpy as np
from PIL import Image
import skimage
from skimage import io
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from .augment import RandAugmentMC, RandAugmentSC, Noise
from .augmix import AugMix, AugMixDatasetSSL


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)
mstar_mean = 0.5
mstar_std = 0.5


# def x_u_split(cfg, labels):
#     label_per_class = cfg.num_labeled // cfg.num_classes
#     labels = np.array(labels)
#     labeled_idx = []
#     # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
#     unlabeled_idx = np.array(range(len(labels)))
#     for i in range(cfg.num_classes):
#         idx = np.where(labels == i)[0]
#         idx = np.random.choice(idx, label_per_class, False)
#         labeled_idx.extend(idx)
#     labeled_idx = np.array(labeled_idx)
#     assert len(labeled_idx) == cfg.num_labeled

#     if cfg.expand_labels or cfg.num_labeled < cfg.batch_size:
#         num_expand_x = math.ceil(cfg.batch_size * cfg.eval_step / cfg.num_labeled)
#         labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
#     np.random.shuffle(labeled_idx)
#     return labeled_idx, unlabeled_idx

def x_u_split(cfg, labels):
    label_per_class = cfg.num_labeled // cfg.num_classes
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = np.array(range(len(labels)))

    for i in range(cfg.num_classes):
        idx = np.where(labels == i)[0]
        if len(idx) == 0:
            raise ValueError(f"No samples found for class {i}. Check your dataset or class distribution.")
        idx = np.random.choice(idx, min(label_per_class, len(idx)), False)
        labeled_idx.extend(idx)

    labeled_idx = np.array(labeled_idx)
    if len(labeled_idx) != cfg.num_labeled:
        print(f"Warning: Number of labeled samples ({len(labeled_idx)}) is less than expected ({cfg.num_labeled}).")
    
    if cfg.expand_labels or cfg.num_labeled < cfg.batch_size:
        num_expand_x = math.ceil(cfg.batch_size * cfg.eval_step / len(labeled_idx))
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


def get_cifar10(cfg, root):
    transform_labeled = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=32, padding=int(32 * 0.125), padding_mode="reflect"
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
        ]
    )
    transform_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
        ]
    )
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(cfg, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True, transform=transform_labeled
    )

    # train_unlabeled_dataset = CIFAR10SSL(
    #     root, train_unlabeled_idxs, train=True,
    #     transform=TransformFixMatchAugMix(mean=cifar10_mean, std=cifar10_std))
    train_unlabeled_dataset = AugMixDatasetSSL(base_dataset, train_unlabeled_idxs)

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False
    )

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(
        self,
        root,
        indexs,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def get_cifar100(cfg, root):
    transform_labeled = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=32, padding=int(32 * 0.125), padding_mode="reflect"
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar100_mean, std=cifar100_std),
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar100_mean, std=cifar100_std),
        ]
    )

    base_dataset = datasets.CIFAR100(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(cfg, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True, transform=transform_labeled
    )

    train_unlabeled_dataset = CIFAR100SSL(
        root,
        train_unlabeled_idxs,
        train=True,
        transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std),
    )

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False
    )

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(
        self,
        root,
        indexs,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def get_MSTAR(cfg, root):
    transform_labeled = transforms.Compose(
        [
            transforms.Resize((cfg.size, cfg.size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=cfg.size, padding=int(cfg.size * 0.125), padding_mode="reflect"
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mstar_mean, std=mstar_std),
        ]
    )

    base_dataset = MSTAR(
        root, train=True, mean=mstar_mean, std=mstar_std, size=cfg.size
    )

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(cfg, base_dataset.targets)

    train_labeled_dataset = MSTARSSL(
        root, train_labeled_idxs, train=True, transform=transform_labeled
    )

    train_unlabeled_dataset = MSTARSSL(
        root,
        train_unlabeled_idxs,
        train=True,
        transform=TransformFixMatchAugMix(mstar_mean, mstar_std, cfg.size),
    )

    test_dataset = MSTAR(
        root, train=False, mean=mstar_mean, std=mstar_std, size=cfg.size
    )

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


class MSTAR(Dataset):
    def __init__(self, root, train, mean, std, size):
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((size, size)),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        self.train = train

        if self.train:
            self.data = datasets.ImageFolder(root + "/train")
        else:
            self.data = datasets.ImageFolder(root + "/test")
        self.img, self.targets = zip(*self.data.imgs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.transforms(
            skimage.util.img_as_ubyte(io.imread(self.img[index], as_gray=True))
        )
        target = self.targets[index]

        return img, target


class MSTARSSL(Dataset):
    def __init__(self, root, indexes, train=True, transform=None):
        self.train = train
        self.transform = transform
        self.indexes = indexes

        if self.train:
            self.data = datasets.ImageFolder(root + "/train")
        else:
            self.data = datasets.ImageFolder(root + "/test")
        train_img, train_label = zip(*self.data.imgs)

        if indexes is not None:
            indexes = indexes.astype(int)
            self.data = np.array(train_img)[indexes]
            self.targets = np.array(train_label)[indexes]

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        img = io.imread(self.data[index], as_gray=True)
        img = Image.fromarray(img)
        img = self.transform(img)
        target = self.targets[index]

        return img, target


def get_sentinel1(cfg, root):
    transform_labeled = transforms.Compose(
        [
            transforms.Resize((cfg.size, cfg.size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=cfg.size, padding=int(cfg.size * 0.125), padding_mode="reflect"
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mstar_mean, std=mstar_std),
        ]
    )

    transform_val = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((cfg.size, cfg.size)),
                transforms.Normalize(mean=mstar_mean, std=mstar_std),
            ]
        )

    base_dataset = Sentinel1(
        root, train=True, mean=mstar_mean, std=mstar_std, size=cfg.size
    )

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(cfg, base_dataset.targets)

    train_labeled_dataset = Sentinel1SSL(
        root, train_labeled_idxs, train=True, transform=transform_labeled
    )
    train_unlabeled_dataset = Sentinel1SSL(
        root,
        train_unlabeled_idxs,
        train=True,
        transform=TransformSelectAugment(mstar_mean, mstar_std, cfg.size, cfg.aug),
    )
    test_dataset = Sentinel1(
        root, train=False, mean=mstar_mean, std=mstar_std, size=cfg.size, transform=transform_val
    )

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


class Sentinel1(Dataset):
    def __init__(self, root, train, mean, std, size=256, transform=None):
        self.root = root
        self.train = train
        self.mean = mean
        self.std = std
        self.size = size
        self.transform = transform

        # 데이터 분할이 필요한 경우 분할 수행
        self._prepare_data()

        # train 또는 test 데이터 로드
        if self.train:
            self.data = datasets.ImageFolder(root + "/train")
        else:
            self.data = datasets.ImageFolder(root + "/test")

        self.img, self.targets = zip(*self.data.imgs)

    def _prepare_data(self):
        """각 클래스별로 all 폴더의 데이터를 train/test로 8:2 비율로 분할하여 저장합니다."""
        train_path = os.path.join(self.root, "train")
        test_path = os.path.join(self.root, "test")
        all_path = os.path.join(self.root, "all")

        # train/test 폴더가 없을 경우에만 분할 수행
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            os.makedirs(train_path, exist_ok=True)
            os.makedirs(test_path, exist_ok=True)

            # all 폴더의 각 클래스 폴더를 대상으로 데이터 분할
            for class_name in os.listdir(all_path):
                class_folder = os.path.join(all_path, class_name)

                # 클래스별 데이터 분할 수행
                if os.path.isdir(class_folder):
                    images = [
                        os.path.join(class_folder, img)
                        for img in os.listdir(class_folder)
                        if img.endswith(".png")
                    ]

                    # 클래스별 train/test 분할 수행
                    train_images, test_images = train_test_split(
                        images, train_size=0.8, random_state=42
                    )

                    # 분할된 이미지를 각각 클래스별로 train/test 폴더에 저장
                    train_class_path = os.path.join(train_path, class_name)
                    test_class_path = os.path.join(test_path, class_name)
                    os.makedirs(train_class_path, exist_ok=True)
                    os.makedirs(test_class_path, exist_ok=True)

                    for img_path in train_images:
                        shutil.copy(img_path, train_class_path)
                    for img_path in test_images:
                        shutil.copy(img_path, test_class_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.transform is not None:
            img = self.transform(
                skimage.util.img_as_ubyte(io.imread(self.img[index], as_gray=True))
            )
        else:
            img = skimage.util.img_as_ubyte(io.imread(self.img[index], as_gray=True))

        target = self.targets[index]

        return img, target


class Sentinel1SSL(Dataset):
    def __init__(self, root, indexes, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train
        self.indexes = indexes

        # train 또는 test 데이터 로드
        if self.train:
            self.data = datasets.ImageFolder(root + "/train")
        else:
            self.data = datasets.ImageFolder(root + "/test")
        train_img, train_label = zip(*self.data.imgs)

        # 인덱스로 필터링
        if indexes is not None:
            indexes = indexes.astype(int)
            self.data = np.array(train_img)[indexes]
            self.targets = np.array(train_label)[indexes]

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        img = io.imread(self.data[index], as_gray=True)
        img = Image.fromarray(img)
        target = self.targets[index]

        if self.transform:
            img = self.transform(img)

        return img, target


def get_sentinel2(cfg, root):
    transform_labeled = transforms.Compose(
        [
            transforms.Resize((cfg.size, cfg.size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=cfg.size, padding=int(cfg.size * 0.125), padding_mode="reflect"
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=normal_mean, std=normal_std),
        ]
    )

    transform_val = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((cfg.size, cfg.size)),
                transforms.Normalize(mean=normal_mean, std=normal_std),
            ]
        )

    base_dataset = Sentinel2(
        root, train=True, mean=normal_mean, std=normal_std, size=cfg.size
    )

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(cfg, base_dataset.targets)

    train_labeled_dataset = Sentinel2SSL(
        root, train_labeled_idxs, train=True, transform=transform_labeled
    )
    train_unlabeled_dataset = Sentinel2SSL(
        root,
        train_unlabeled_idxs,
        train=True,
        transform=TransformFixMatch(normal_mean, normal_std, cfg.size),
    )
    test_dataset = Sentinel2(
        root, train=False, mean=normal_mean, std=normal_std, size=cfg.size, transform=transform_val
    )

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


class Sentinel2(Dataset):
    def __init__(self, root, train, mean, std, size=256, transform=None):
        self.root = root
        self.train = train
        self.mean = mean
        self.std = std
        self.size = size
        self.transform = transform

        # 데이터 분할이 필요한 경우 분할 수행
        self._prepare_data()

        # train 또는 test 데이터 로드
        if self.train:
            self.data = datasets.ImageFolder(root + "/train")
        else:
            self.data = datasets.ImageFolder(root + "/test")

        self.img, self.targets = zip(*self.data.imgs)

    def _prepare_data(self):
        """각 클래스별로 all 폴더의 데이터를 train/test로 8:2 비율로 분할하여 저장합니다."""
        train_path = os.path.join(self.root, "train")
        test_path = os.path.join(self.root, "test")
        all_path = os.path.join(self.root, "all")

        # train/test 폴더가 없을 경우에만 분할 수행
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            os.makedirs(train_path, exist_ok=True)
            os.makedirs(test_path, exist_ok=True)

            # all 폴더의 각 클래스 폴더를 대상으로 데이터 분할
            for class_name in os.listdir(all_path):
                class_folder = os.path.join(all_path, class_name)

                # 클래스별 데이터 분할 수행
                if os.path.isdir(class_folder):
                    images = [
                        os.path.join(class_folder, img)
                        for img in os.listdir(class_folder)
                        if img.endswith(".png")
                    ]

                    # 클래스별 train/test 분할 수행
                    train_images, test_images = train_test_split(
                        images, train_size=0.8, random_state=42
                    )

                    # 분할된 이미지를 각각 클래스별로 train/test 폴더에 저장
                    train_class_path = os.path.join(train_path, class_name)
                    test_class_path = os.path.join(test_path, class_name)
                    os.makedirs(train_class_path, exist_ok=True)
                    os.makedirs(test_class_path, exist_ok=True)

                    for img_path in train_images:
                        shutil.copy(img_path, train_class_path)
                    for img_path in test_images:
                        shutil.copy(img_path, test_class_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.transform is not None:
            img = self.transform(
                skimage.util.img_as_ubyte(io.imread(self.img[index], as_gray=False))
            )
        else:
            img = skimage.util.img_as_ubyte(io.imread(self.img[index], as_gray=False))

        target = self.targets[index]

        return img, target


class Sentinel2SSL(Dataset):
    def __init__(self, root, indexes, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train
        self.indexes = indexes

        # train 또는 test 데이터 로드
        if self.train:
            self.data = datasets.ImageFolder(root + "/train")
        else:
            self.data = datasets.ImageFolder(root + "/test")
        train_img, train_label = zip(*self.data.imgs)

        # 인덱스로 필터링
        if indexes is not None:
            indexes = indexes.astype(int)
            self.data = np.array(train_img)[indexes]
            self.targets = np.array(train_label)[indexes]

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        img = io.imread(self.data[index], as_gray=False)
        img = Image.fromarray(img)
        target = self.targets[index]

        if self.transform:
            img = self.transform(img)

        return img, target


class TransformFixMatch(object):
    def __init__(self, mean, std, size):
        self.weak = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=size, padding=int(size * 0.125), padding_mode="reflect"
                ),
            ]
        )
        self.strong = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=size, padding=int(size * 0.125), padding_mode="reflect"
                ),
                RandAugmentMC(n=1, m=10),
            ]
        )
        self.normalize = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class TransformSelectAugment(object):
    def __init__(self, mean, std, size, aug):
        self.weak = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=size, padding=int(size * 0.125), padding_mode="reflect"
                ),
            ]
        )
        self.strong = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=size, padding=int(size * 0.125), padding_mode="reflect"
                ),
                RandAugmentSC(name=aug, m=10),
            ]
        )
        self.normalize = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class TransformFixMatchAugMix(object):
    def __init__(self, mean, std, size):
        self.base = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=size, padding=int(size * 0.125), padding_mode="reflect"
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        self.weak = self.base
        self.strong1 = AugMix(transform=self.base)
        self.strong2 = AugMix(transform=self.base)

    def __call__(self, x):
        weak = self.weak(x)
        strong1 = self.strong1(x)
        strong2 = self.strong2(x)
        return weak, strong1, strong2


class TransformWithNoise(object):
    def __init__(self, size):
        self.weak = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.Resize((size, size))]
        )
        self.strong = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize((size, size)),
                # for FGSM
                # RandAugmentMC(n=1, m=10)
            ]
        )
        self.normalize = transforms.Compose([transforms.ToTensor()])

    def __call__(self, x):
        # x = Noise(x)
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


DATASET_GETTERS = {
    "cifar10": get_cifar10,
    "cifar100": get_cifar100,
    "MSTAR": get_MSTAR,
    "sentinel1": get_sentinel1,
    "sentinel2": get_sentinel2,
}

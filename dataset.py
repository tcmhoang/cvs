import torch
import torchvision
from torch._prims_common import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, transforms

from dio import List, Tuple


def get_img_train_transform(
    crop_sz: int, norm_mean: List[float], norm_std: List[float]
) -> Compose:
    return torchvision.transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=crop_sz,
                scale=(0.85, 1.0),
                ratio=(0.95, 1.05),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(45),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,  # camera sensor differences
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        ]
    )


def get_img_test_transform(
    sqr_sz: int, crop_sz: int, norm_mean: List[float], norm_std: List[float]
) -> Compose:
    return transforms.Compose(
        [
            transforms.Resize((sqr_sz, sqr_sz)),
            transforms.CenterCrop(crop_sz),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ]
    )


def get_img_sets(
    train_dir_w_transfomer: Tuple[str, Compose],
    test_dir_w_transformer: Tuple[str, Compose],
) -> Tuple[ImageFolder, ImageFolder]:
    return (
        torchvision.datasets.ImageFolder(
            root=train_dir_w_transfomer[0], transform=train_dir_w_transfomer[1]
        ),
        torchvision.datasets.ImageFolder(
            root=test_dir_w_transformer[0], transform=test_dir_w_transformer[1]
        ),
    )


def get_test_loaders(
    test_set_w_batch_sz: Tuple[ImageFolder, int],
    num_workers: int,
) -> DataLoader[Tensor]:
    return DataLoader(
        test_set_w_batch_sz[0],
        batch_size=test_set_w_batch_sz[1],
        shuffle=False,
        num_workers=num_workers,
    )


def calc_mean_and_stdev(data_dir: str) -> Tuple[List[float], List[float]]:
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    tensor_dataset = ImageFolder(root=data_dir, transform=transform)

    loader = DataLoader(tensor_dataset, batch_size=32, num_workers=0, shuffle=False)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)

        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images
    return mean.tolist(), std.tolist()

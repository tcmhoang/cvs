from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, transforms
from dio import List, Tuple
import torchvision


def get_img_train_transform(
    sqr_sz: int, crop_sz: int, norm_mean: List[float], norm_std: List[float]
) -> Compose:
    return torchvision.transforms.Compose(
        [
            transforms.Resize((sqr_sz, sqr_sz)),
            transforms.CenterCrop(crop_sz),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
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
) -> DataLoader:
    return DataLoader(
        test_set_w_batch_sz[0],
        batch_size=test_set_w_batch_sz[1],
        shuffle=False,
        num_workers=num_workers,
    )

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms 



def get_dataloaders(data_dir="./data", batch_size=128, num_workers=4):

    # Mean & Std of CIFAR10 dataset --> precomputed from online
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2470, 0.2435, 0.2616)



    # ========================= #
    # Define transformations
    # ========================= #

    train_transform = transforms.Compose([
        # Data augmentation to increase dataset size and variety --> to improve generalization 
        transforms.RandomHorizontalFlip(), 
        transforms.RandomCrop(size=32, padding=4), 

        # Apply normalization
        transforms.ToTensor(), 
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])


    # ========================= #
    # Load datasets -> CIFAR10
    # ========================= #

    train_dataset = datasets.CIFAR10(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=test_transform
    )


    # ========================= #
    # Setup DataLoaders
    # ========================= #

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4, 
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4, 
        persistent_workers=True
    )

    return train_loader, test_loader

    

    


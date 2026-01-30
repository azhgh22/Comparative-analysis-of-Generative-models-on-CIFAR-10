# Dataset loading and preprocessing for CIFAR-10

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
from torchvision import datasets, transforms

def load_cifar10(
        batch_size: int=128,
        data_dir: str='./data',
        normalize_inputs: bool=False,
        pin_memory: bool=False,
        num_workers: int=0) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ] if normalize_inputs else [
        transforms.ToTensor()
    ]
    transform = transforms.Compose([*transform_list])
    
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
    
    return train_loader, test_loader
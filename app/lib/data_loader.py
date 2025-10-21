import torch
from torchvision import datasets, transforms


def get_data_loader(data_dir: str, batch_size: int, train: bool):
    """
    :param data_dir: The download path of the dataset.
    :param train: Training or not
    """
    # TODO: create the data loader
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    return loader

from torch.utils.data import Dataset
from torchvision import datasets, transforms


def get_dataset(
    dataset_name: str, img_size: int, data_root: str = "./temp_data"
) -> Dataset:
    is_gray = dataset_name in ["MNIST", "FashionMNIST"]

    transform_list = [
        transforms.Resize((img_size, img_size)),
    ]

    if is_gray:
        transform_list.append(transforms.Grayscale(num_output_channels=3))

    transform_list.extend(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    transform = transforms.Compose(transform_list)

    dataset_map = {
        "MNIST": datasets.MNIST,
        "CIFAR10": datasets.CIFAR10,
        "FashionMNIST": datasets.FashionMNIST,
        "SVHN": datasets.SVHN,
    }

    if dataset_name not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_class = dataset_map[dataset_name]
    
    if dataset_name == "SVHN":
        train_dataset = dataset_class(root=data_root, split="train", download=True, transform=transform)
        # For simplicity, we'll return the train split for both, as the task is continual learning
        return train_dataset
    else:
        train_dataset = dataset_class(root=data_root, train=True, download=True, transform=transform)
        return train_dataset

from torchvision import datasets, transforms


def get_dataset(
    dataset_name: str, batch_size: int, img_size: int, data_root: str = "./temp_data"
) -> dict:
    """
    Loads and transforms a dataset based on its name.
    Args:
        dataset_name (str): Name of the dataset (e.g., 'MNIST', 'CIFAR10').
        batch_size (int): Batch size for the DataLoader.
        img_size (int): Target image size for transformation.
        data_root (str): Root directory where datasets will be downloaded/loaded from.
    Returns:
        dict: A dictionary containing the training and test DataLoaders.
    """
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

    if dataset_name == "MNIST":
        train_dataset = datasets.MNIST(
            root=data_root, train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root=data_root, train=False, download=True, transform=transform
        )
    elif dataset_name == "CIFAR10":
        train_dataset = datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=transform
        )
    elif dataset_name == "FashionMNIST":
        train_dataset = datasets.FashionMNIST(
            root=data_root, train=True, download=True, transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            root=data_root, train=False, download=True, transform=transform
        )
    elif dataset_name == "SVHN":
        train_dataset = datasets.SVHN(
            root=data_root, split="train", download=True, transform=transform
        )
        test_dataset = datasets.SVHN(
            root=data_root, split="test", download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return {"train": train_dataset, "test": test_dataset}

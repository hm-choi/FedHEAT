from utils.registry import Registry
from torchvision.datasets import CIFAR10, CIFAR100
from .tiny_imagenet import TinyImageNet
from torchvision import transforms
import yaml

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for datasets
"""
DATASET_REGISTRY.register(CIFAR10)
DATASET_REGISTRY.register(CIFAR100)
DATASET_REGISTRY.register(TinyImageNet)

__all__ = ['build_dataset', 'build_datasets']


def get_transform(train, config):
    normalize = transforms.Normalize(config['mean'],
                                        config['std'])
    imsize = config['imsize']
    if train:
        transform = transforms.Compose(
            [transforms.RandomRotation(10),
                transforms.RandomCrop(imsize, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
                ])
    else:
        transform = transforms.Compose(
            [transforms.CenterCrop(imsize),
                transforms.ToTensor(),
                normalize])

    return transform


def build_dataset(args, train=True):
    if args.verbose and train == True:
        print(DATASET_REGISTRY)

    download = args.dataset.download if args.dataset.get('download') else False

    with open('datasets/configs.yaml', 'r') as f:
        dataset_config = yaml.safe_load(f)[args.dataset.name]
    transform = get_transform(train, dataset_config)
    dataset = DATASET_REGISTRY.get(args.dataset.name)(root=args.dataset.path, download=download, train=train, transform=transform) if len(args.dataset.path) > 0 else None

    return dataset


def build_datasets(args):
    train_dataset = build_dataset(args, train=True)
    test_dataset = build_dataset(args, train=False)
    
    datasets = {
        "train": train_dataset,
        "test": test_dataset,
    }

    # --------------------------------
    # statistics
    # --------------------------------
    num_train_samples = len(train_dataset)
    num_test_samples  = len(test_dataset)

    print("===== Dataset Statistics =====")
    print(f"Total train samples    : {num_train_samples}")
    print(f"Total test samples     : {num_test_samples}")
    print("==============================")

    return datasets
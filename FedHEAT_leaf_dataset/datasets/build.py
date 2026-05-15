# 코드 확인 완료
from utils.registry import Registry
from torchvision import transforms
from torchvision.transforms import Normalize
import yaml

from datasets.leaf_femnist import LEAF_FEMNIST
from datasets.leaf_shakespeare import LEAF_SHAKESPEARE
from datasets.leaf_celeba import LEAF_CELEBA

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for datasets
"""
DATASET_REGISTRY.register(LEAF_FEMNIST)
DATASET_REGISTRY.register(LEAF_SHAKESPEARE)
DATASET_REGISTRY.register(LEAF_CELEBA)

__all__ = ['build_dataset', 'build_datasets']


def get_transform(args):
    if 'LEAF_FEMNIST' in args.dataset.name:
        transform = None
    elif 'LEAF_CELEBA' in args.dataset.name:
        transform = transforms.Compose([
            transforms.Resize(size=(84, 84)),
            transforms.ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    elif 'LEAF_SHAKESPEARE' in args.dataset.name:
        transform = None
    else:
        raise ValueError(f"Unsupported LEAF dataset: {args.dataset.name}")
    
    return transform


def build_dataset(args, train=True):  
    if args.verbose and train == True:
        print(DATASET_REGISTRY)

    if 'LEAF' not in args.dataset.name:
        raise ValueError(f"Unsupported dataset in LEAF code: {args.dataset.name}")

    download = args.dataset.download if args.dataset.get('download') else False

    transform = get_transform(args)
    dataset_cls = DATASET_REGISTRY.get(args.dataset.name)

    return dataset_cls(
        root=args.dataset.path,
        download=download,
        train=train,
        transform=transform,
        seed=args.seed if args.get("seed") is not None else 42,
    )


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
    num_train_samples = sum(
        len(train_dataset.get_client_dataset(client_idx))
        for client_idx in train_dataset.clients
    )
    num_test_samples  = len(test_dataset)

    print("===== Dataset Statistics =====")
    print(f"Total clients           : {len(train_dataset.clients)}")
    print(f"Total train samples     : {num_train_samples}")
    print(f"Total test samples      : {num_test_samples}")
    print("==============================")

    return datasets
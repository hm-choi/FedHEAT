import json
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


IMAGE_SIZE = 84


class ClientTensorDataset(Dataset):
    def __init__(
        self,
        xs_raw,
        ys_raw,
        images_dir,
        transform=None,
    ):
        self.xs_raw = xs_raw
        self.ys_raw = ys_raw

        self.images_dir = Path(images_dir)
        self.transform = transform

        self.tensorized = False

        self.xs = None
        self.ys = None

        self.class_dict = dict(
            Counter(int(y) for y in ys_raw)
        )

    def __len__(self):
        return len(self.ys_raw)

    def _process_x(self, img_name):

        img = Image.open(
            self.images_dir / img_name
        ).convert("RGB")

        # 핵심:
        # tensor화할 때 transform 적용
        if self.transform is not None:
            x = self.transform(img)

        else:
            img = img.resize((84,84))

            x = torch.tensor(
                np.array(img),
                dtype=torch.float32
            ).permute(2,0,1)

        return x

    def tensorize(self):

        if self.tensorized:
            return

        self.xs = [
            self._process_x(x)
            for x in self.xs_raw
        ]

        self.ys = [
            int(y)
            for y in self.ys_raw
        ]

        self.tensorized = True

    def __getitem__(self, idx):

        if self.tensorized:
            return (
                self.xs[idx],
                self.ys[idx]
            )

        # lazy mode도 동일하게 transform 적용
        return (
            self._process_x(
                self.xs_raw[idx]
            ),
            int(
                self.ys_raw[idx]
            )
        )


class LEAF_CELEBA(Dataset):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        download=False,
        seed=42,
        images_dir=None,
    ):
        self.root = Path(root)
        self.train = train
        self.transform = transform
        self.download = download
        self.seed = seed

        if images_dir is None:
            # 필요하면 실제 이미지 경로에 맞게 수정
            self.images_dir = self.root / "data" / "raw" / "img_align_celeba"
        else:
            self.images_dir = Path(images_dir)

        split_dir = self.root / ("train" if train else "test")
        self.files = sorted(split_dir.glob("*.json"))

        if len(self.files) == 0:
            raise FileNotFoundError(f"No json files found in {split_dir}")

        self.classes = None

        if self.train:
            self.clients = []
            self.client_data = {}
            self.original_client_ids = []

            self._build_train_federated_dataset()
            self.classes = self._infer_classes_from_client_data()
        else:
            self.data = []
            self.targets = []

            self._build_test_global_dataset()
            self.classes = self._infer_classes_from_targets()

    def _load_json(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def _infer_classes_from_client_data(self):
        label_set = set()

        for client_dict in self.client_data.values():
            label_set.update(int(y) for y in client_dict["y_raw"])

        max_label = max(label_set)
        return list(range(max_label + 1))

    def _infer_classes_from_targets(self):
        max_label = max(self.targets)
        return list(range(max_label + 1))

    def _process_x(self, img_name):
        img_path = self.images_dir / img_name

        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            x = self.transform(img)
        else:
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            x = torch.tensor(
                np.array(img),
                dtype=torch.float32
            ).permute(2, 0, 1)

        return x

    def _build_train_federated_dataset(self):
        tmp_users = []
        tmp_user_data = {}

        for fp in self.files:
            data = self._load_json(fp)
            users = data["users"]
            user_data = data["user_data"]

            for user in users:
                tmp_users.append(user)
                tmp_user_data[user] = user_data[user]

        selected_users = tmp_users

        for new_idx, user in enumerate(selected_users):
            raw_x = tmp_user_data[user]["x"]  # image filenames
            raw_y = tmp_user_data[user]["y"]

            self.clients.append(new_idx)
            self.original_client_ids.append(user)

            # 중요: train에서는 이미지를 읽거나 tensor로 바꾸지 않음
            self.client_data[new_idx] = {
                "x_raw": list(raw_x),
                "y_raw": list(raw_y),
            }

    def _build_test_global_dataset(self):
        for fp in self.files:
            data = self._load_json(fp)
            users = data["users"]
            user_data = data["user_data"]

            for user in users:
                raw_x = user_data[user]["x"]
                raw_y = user_data[user]["y"]

                for x, y in zip(raw_x, raw_y):
                    self.data.append(self._process_x(x))
                    self.targets.append(int(y))

    def get_client_dataset(self, client_idx):
        client_dict = self.client_data[client_idx]

        return ClientTensorDataset(
            client_dict["x_raw"],
            client_dict["y_raw"],
            images_dir=self.images_dir,
            transform=self.transform,
        )

    def __len__(self):
        if self.train:
            return len(self.clients)

        return len(self.targets)

    def __getitem__(self, idx):
        if self.train:
            raise RuntimeError(
                "Train split is federated. Use get_client_dataset(client_idx) instead of __getitem__."
            )

        return self.data[idx], self.targets[idx]
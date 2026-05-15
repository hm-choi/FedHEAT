import json
import random
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from collections import Counter

# train/test 두 Dataset이 같은 root로 만들어질 때, 합집합 vocab 스캔(JSON 전체 파싱)을 한 번만 수행
_VOCAB_UNION_CACHE: Dict[str, List[str]] = {}

class ClientTensorDataset(Dataset):
    def __init__(self, xs_raw, ys_raw, class_to_idx, raws=None):
        self.xs_raw = xs_raw
        self.ys_raw = ys_raw
        self.class_to_idx = class_to_idx
        self.raws = raws

        # label 통계용 (tensor 필요 없음)
        self.class_dict = dict(
            Counter(self._label_to_int(y) for y in ys_raw)
        )

        # optional cache (한번 변환 후 재사용)
        self.tensorized = False
        self.xs = None
        self.ys = None

    def _encode_sequence(self, x):
        if torch.is_tensor(x):
            return x.long()

        if isinstance(x, str):
            return torch.tensor(
                [self.class_to_idx[ch] for ch in x],
                dtype=torch.long
            )

        if isinstance(x, list):
            return torch.tensor(x, dtype=torch.long)

        return torch.tensor(list(x), dtype=torch.long)

    def _encode_target(self, y):
        if torch.is_tensor(y):
            return y.long().view(-1)[0]

        if isinstance(y, str):
            return torch.tensor(
                self.class_to_idx[y],
                dtype=torch.long
            )

        return torch.tensor(int(y), dtype=torch.long)

    def _label_to_int(self, y):
        if isinstance(y, str):
            return self.class_to_idx[y]
        return int(y)

    # 필요할 때 전체 tensor화
    def tensorize(self):
        if self.tensorized:
            return

        self.xs = [
            self._encode_sequence(x)
            for x in self.xs_raw
        ]

        self.ys = [
            self._encode_target(y)
            for y in self.ys_raw
        ]

        self.tensorized = True

    def __len__(self):
        return len(self.ys_raw)

    def __getitem__(self, idx):

        # lazy conversion 방식 1:
        # 샘플 접근할 때마다 변환
        if not self.tensorized:
            x = self._encode_sequence(
                self.xs_raw[idx]
            )
            y = self._encode_target(
                self.ys_raw[idx]
            )
        else:
            x = self.xs[idx]
            y = self.ys[idx]

        if self.raws is None:
            return x, y

        return x, y, self.raws[idx]


class LEAF_SHAKESPEARE(Dataset):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        num_clients=None,
        seed=42,
        keep_raw=False,
        vocab=None,
    ):
        self.root = Path(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.seed = seed
        self.keep_raw = keep_raw

        split_dir = self.root / ("train" if train else "test")
        self.files = sorted(split_dir.glob("*.json"))

        if len(self.files) == 0:
            raise FileNotFoundError(f"No json files found in {split_dir}")

        self.clients = []
        self.client_data = {}
        self.original_client_ids = []
        self.client_hierarchies = {}
        self.data = []
        self.targets = []
        self.raws = []
        self.targets_raw = []

        # One global character vocabulary for train and test (same class_to_idx for both splits).
        if vocab is not None:
            self.classes = list(vocab)
        else:
            self.classes = self._infer_classes_union_train_test()

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = {i: c for i, c in enumerate(self.classes)}

        if self.train:
            self._build_train_federated_dataset(num_clients=num_clients)
        else:
            self._build_test_global_dataset()

        self._encode_sequences()

    def _load_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _infer_classes_union_train_test(self):
        """All chars appearing in train OR test JSON (same mapping for both splits)."""
        cache_key = str(self.root.resolve())
        if cache_key in _VOCAB_UNION_CACHE:
            return list(_VOCAB_UNION_CACHE[cache_key])

        char_set = set()
        for split_name in ("train", "test"):
            split_dir = self.root / split_name
            if not split_dir.is_dir():
                continue
            for fp in sorted(split_dir.glob("*.json")):
                data = self._load_json(fp)
                for user in data["users"]:
                    entry = data["user_data"][user]
                    for x in entry["x"]:
                        char_set.update(x)
                    for y in entry["y"]:
                        char_set.add(y)
        if not char_set:
            raise ValueError(
                f"No characters found under {self.root}/train and {self.root}/test JSON files."
            )
        classes = sorted(char_set)
        _VOCAB_UNION_CACHE[cache_key] = classes
        return classes

    def _build_train_federated_dataset(self, num_clients=None):
        tmp_users = []
        tmp_user_data = {}
        tmp_hierarchies = {}

        for fp in self.files:
            data = self._load_json(fp)

            users = data["users"]
            user_data = data["user_data"]
            hierarchies = data.get("hierarchies", None)

            for i, user in enumerate(users):
                tmp_users.append(user)
                tmp_user_data[user] = user_data[user]
                tmp_hierarchies[user] = hierarchies[i] if hierarchies is not None else None

        if num_clients is not None and num_clients < len(tmp_users):
            rng = random.Random(self.seed)
            selected_users = rng.sample(tmp_users, num_clients)
        else:
            selected_users = tmp_users

        for new_idx, user in enumerate(selected_users):
            entry = tmp_user_data[user]

            raw_x = entry["x"]
            raw_y = entry["y"]
            raw_raw = entry.get("raw", None)

            self.clients.append(new_idx)
            self.original_client_ids.append(user)
            self.client_hierarchies[new_idx] = tmp_hierarchies[user]

            self.client_data[new_idx] = {
                "x": list(raw_x),
                "y": list(raw_y),
                "x_raw": list(raw_x),
                "y_raw": raw_y,
                "raw": raw_raw if self.keep_raw else None,
            }

    def _build_test_global_dataset(self):
        for fp in self.files:
            data = self._load_json(fp)
            users = data["users"]
            user_data = data["user_data"]

            for user in users:
                entry = user_data[user]

                raw_x = entry["x"]
                raw_y = entry["y"]
                raw_raw = entry.get("raw", None)

                for i, (x, y) in enumerate(zip(raw_x, raw_y)):
                    self.data.append(x)
                    self.targets.append(y)
                    self.targets_raw.append(y)

                    if self.keep_raw:
                        self.raws.append(raw_raw[i] if raw_raw is not None else None)

    def get_client_dataset(self, client_idx):
        client_dict = self.client_data[client_idx]
        return ClientTensorDataset(
            client_dict["x_raw"],
            client_dict["y_raw"],
            self.class_to_idx,
            client_dict["raw"] if self.keep_raw else None,
        )

    def _encode_sequence(self, x):
        if torch.is_tensor(x):
            return x.long()
        if isinstance(x, str):
            return torch.tensor([self.class_to_idx[ch] for ch in x], dtype=torch.long)
        if isinstance(x, list):
            return torch.tensor(x, dtype=torch.long)
        return torch.tensor(list(x), dtype=torch.long)

    def _encode_target(self, y):
        if torch.is_tensor(y):
            return y.long().view(-1)[0]
        if isinstance(y, str):
            return torch.tensor(self.class_to_idx[y], dtype=torch.long)
        return torch.tensor(int(y), dtype=torch.long)

    def _encode_sequences(self):
        if self.train:
            return
        else:
            self.data = [self._encode_sequence(x) for x in self.data]
            self.targets = [self._encode_target(y) for y in self.targets]

    def __len__(self):
        if self.train:
            return len(self.clients)
        return len(self.targets)

    def __getitem__(self, idx):
        if self.train:
            raise RuntimeError(
                "Train split is federated. Use get_client_dataset(client_idx) instead of __getitem__."
            )

        if self.keep_raw:
            return self.data[idx], self.targets[idx], self.raws[idx]
        return self.data[idx], self.targets[idx]
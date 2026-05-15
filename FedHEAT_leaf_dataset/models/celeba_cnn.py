import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from models.build import ENCODER_REGISTRY


IMAGE_SIZE = 84


class CelebAClientDataset(Dataset):
    def __init__(self, img_names, labels, images_dir, transform=None):
        self.img_names = img_names
        self.labels = labels
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        label = int(self.labels[idx])

        img = Image.open(self.images_dir / img_name).convert("RGB")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

        if self.transform is not None:
            x = self.transform(img)
        else:
            x = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0

        y = torch.tensor(label, dtype=torch.long)
        return x, y


@ENCODER_REGISTRY.register()
class CELEBA_CNN(nn.Module):
    def __init__(self, args=None, num_classes=2, l2_norm=False, num_groups=8, **kwargs):
        super(CELEBA_CNN, self).__init__()

        self.block1 = self._make_block(3, 32, num_groups)
        self.block2 = self._make_block(32, 32, num_groups)
        self.block3 = self._make_block(32, 32, num_groups)
        self.block4 = self._make_block(32, 32, num_groups)

        # 84 -> 42 -> 21 -> 11 -> 6 with ceil_mode=True
        self.fc = nn.Linear(32 * 6 * 6, num_classes)

        self.num_layers = 4
        self.l2_norm = l2_norm

    def _make_block(self, in_channels, out_channels, num_groups):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        )

    def forward(self, x, mlb_level=None):
        x = self.block1(x)
        layer0 = x

        x = self.block2(x)
        layer1 = x

        x = self.block3(x)
        layer2 = x

        x = self.block4(x)
        layer3 = x

        feature = x.view(x.size(0), -1)

        if self.l2_norm:
            feature = F.normalize(feature, p=2, dim=1)

        logit = self.fc(feature)

        return {
            "layer0": layer0,
            "layer1": layer1,
            "layer2": layer2,
            "layer3": layer3,
            "feature": feature,
            "logit": logit,
        }
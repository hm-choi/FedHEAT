import torch
import torch.nn as nn
import torch.nn.functional as F
from models.build import ENCODER_REGISTRY


@ENCODER_REGISTRY.register()
class FEMNIST_CNN(nn.Module):
    def __init__(self, args, num_classes=62, l2_norm=False, **kwargs):
        super(FEMNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 2048)
        self.fc2 = nn.Linear(2048, num_classes)

        self.num_layers = 2

    def forward(self, x, mlb_level=None):
        x = self.pool(F.relu(self.conv1(x)))
        layer0 = x

        x = self.pool(F.relu(self.conv2(x)))
        layer1 = x

        x = x.view(x.size(0), -1)
        feature = F.relu(self.fc1(x))
        logit = self.fc2(feature)

        return {
            "layer0": layer0,
            "layer1": layer1,
            "feature": feature,
            "logit": logit,
        }
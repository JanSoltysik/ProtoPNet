"""
LeNet-5 implementation in PyTorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self, n_classes=2):
        super(LeNet5, self).__init__()
        self.kernel_sizes = [3]
        self.strides = [1]
        self.paddings = [1]

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=3, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=12000, out_features=128 * 4),
            nn.Linear(in_features=128 * 4, out_features=n_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return probs

    def conv_info(self):
        return self.kernel_sizes, self.strides, self.paddings


def lenet5_features(**kwargs):
    return LeNet5()

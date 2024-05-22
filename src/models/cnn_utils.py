import torch
from torch import nn
import torchvision.transforms as transforms


class ToFloat(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = x.to(torch.float32)
        return x


class FixShape(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, img):
        if len(img.shape) != 4 and len(img.shape) != 3:
            raise Exception(f"error: {len(img.shape)} should be 4 but got: x.shape={img.shape}")
        if len(img.shape) == 4:
            if img.shape[1] != 3:
                img = img.transpose(1, -1).transpose(2, 3)

        if len(img.shape) == 3:
            if img.shape[1] != 3:
                img = img.transpose(0, -1).transpose(1, 2)

        return img.to(torch.uint8)


test_transform = transforms.Compose([
    FixShape(),
    transforms.Resize(224),
    ToFloat()
])
train_transform = transforms.Compose([
    FixShape(),
    transforms.Resize(224),
    transforms.RandAugment(),
    ToFloat()
])

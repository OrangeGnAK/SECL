import torch
import os
import rasterio
import numpy as np
import albumentations as A


class Sen1Floods11_transform:
    def __init__(self, mean, std):
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
            A.ToTensorV2()
        ])

    def __call__(self, img, mask):
        aug_data = self.transform(image=img, mask=mask)
        return aug_data['image'], aug_data['mask']


class Cityscape_transform:
    def __init__(self, mean, std):
        self.transform = A.Compose([
            A.Resize(height=512, width=1024),
            A.RandomCrop(height=512,width=512),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.HueSaturationValue(p=0.2),
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
            A.ToTensorV2()
        ])

    def __call__(self, img, mask):
        aug_data = self.transform(image=img, mask=mask)
        return aug_data['image'], aug_data['mask']

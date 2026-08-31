import torch
import os
import rasterio
import numpy as np
import albumentations as A


class Sen1Floods11_transform:
    def __init__(self, mean, std):
        self.transform = A.Compose([
            # A.RandomCrop(height=480,width=480),
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
            A.ToTensorV2()
        ])

    def __call__(self, img, mask):
        aug_data = self.transform(image=img, mask=mask)
        return aug_data['image'], aug_data['mask']

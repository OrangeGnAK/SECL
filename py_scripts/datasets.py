import torch
import os
import rasterio
import numpy as np
import albumentations as A

class Sen1Floods11_DS(torch.utils.data.Dataset):
    def __init__(self, s1_dir, s2_dir, label_dir, transform=None):
        names = [x.removesuffix('_S1Hand.tif') for x in os.listdir(s1_dir)]
        self.names = sorted(names)
        self.s1_dir = s1_dir
        self.s2_dir = s2_dir
        self.label_dir = label_dir
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        id_name = self.names[idx]
        s1_path = os.path.join(self.s1_dir, id_name + '_S1Hand.tif')
        s2_path = os.path.join(self.s2_dir, id_name + '_S2Hand.tif')
        label_path = os.path.join(self.label_dir, id_name+ '_LabelHand.tif')

        with rasterio.open(s2_path) as src:
            # (RGB + NIR)
            s2_data = src.read((4, 3, 2, 8)).astype(np.float32)
            
        with rasterio.open(s1_path) as src:
            # (VV + VH)
            s1_data = src.read((1, 2)).astype(np.float32)

        with rasterio.open(label_path) as src:
            # (VV + VH)
            label_data = src.read(1).astype(np.int64)

        full_img = np.transpose(
            np.concatenate([s2_data, s1_data], axis=0), # (6, 128, 128)
            (1,2,0))

        if self.transform:
            return self.transform(full_img, label_data)
            
        return torch.from_numpy(full_img), torch.from_numpy(label_data)

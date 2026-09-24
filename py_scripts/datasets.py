import torch
import os
import rasterio
import numpy as np
import albumentations as A
import cv2 as cv

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
        else:
            # print(full_img.shape)
            full_img = torch.from_numpy(full_img).permute(2,0,1).to(torch.float32)
            # print(full_img.shape)
            label_data = torch.from_numpy(label_data).to(torch.long)
            
        return full_img, label_data

# TODO: i really need to think what to do with the loss to deal with disbalance of the classes on Cityscapes...
class Cityscapes_DS(torch.utils.data.Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        super().__init__()
        
        img_dir = os.path.join(root_dir, 'Cityscape Dataset', 'leftImg8bit', mode)
        lbl_dir = os.path.join(root_dir, 'Fine Annotations', 'gtFine', mode)
        
        names = self.get_names(img_dir)

        self.id_to_trainid = np.array([
            -1, -1, -1, -1, -1, -1, -1,  0,  1, -1, -1,  2,  3,  4, -1, -1, -1,
             5, -1,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, -1, -1, 16, 17, 18
        ], dtype=np.int64)
        
        self.names = sorted(names)
        self.img_suffix = "_leftImg8bit.png"
        self.lbl_suffix = "_gtFine_labelIds.png"
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.transform = transform

    def get_names(self, img_dir):
        cities = os.listdir(img_dir)
        names = []
        for city in cities:
            names += [s.removesuffix('_leftImg8bit.png') for s in os.listdir(os.path.join(img_dir, city))]
        return names
        
    
    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):

        name = self.names[idx]
        cityname = name.split('_')[0]
        
        img_path = os.path.join(self.img_dir, cityname, name + self.img_suffix)
        lbl_path = os.path.join(self.lbl_dir, cityname, name + self.lbl_suffix)

        img = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB)
        lbl = cv.imread(lbl_path, cv.IMREAD_GRAYSCALE)

        lbl = np.where(lbl < len(self.id_to_trainid), lbl, 0)
        lbl = self.id_to_trainid[lbl]

        if self.transform:
            return self.transform(img, lbl)
        else:
            img = torch.from_numpy(img).permute(2,0,1).to(torch.float32)
            lbl = torch.from_numpy(lbl).to(torch.long)
            
        return img, lbl

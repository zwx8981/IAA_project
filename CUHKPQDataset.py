#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 13:50:33 2018

@author: congbaobao
"""
import os
import numpy as np
from PIL import Image
from torch.utils import data

class CUHKPQDataset(data.Dataset):
    """AVA dataset
    Args:
        csv_file: a 11-column csv_file, column one contains the names of image files, column 2-11 contains the empiricial distributions of ratings
        root_dir: directory to the images
        transform: preprocessing and augmentation of the training images
    """

    def __init__(self, csv_file, root_dir,transform=None):
        self.annotations = np.loadtxt(csv_file,'str') # 'int'
        #self.annotations = self.annotations[0:8845,:]
        self.root_dir = root_dir
        self.transform = transform
#         self.style_ann = np.loadtxt(style_file,'int')

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.annotations[idx, 0]))
        image = Image.open(img_name)
        image = image.convert("RGB")
        
        label = self.annotations[idx][1]
        
        label = int(label)
        #sample = {'image': image, 'annotations': label}

        if self.transform:
            image = self.transform(image)
            #sample['image'] = self.transform(sample['image'])

        return image, label

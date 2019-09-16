#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 15:45:57 2018

@author: qsyang
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
#from PIL import ImageFile
from torch.utils import data
import scipy.io as scio
class AVADataset(data.Dataset):
    """AVA dataset
    Args:
        csv_file: a 11-column csv_file, column one contains the names of image files, column 2-11 contains the empiricial distributions of ratings
        root_dir: directory to the images
        transform: preprocessing and augmentation of the training images
    """

    def __init__(self, mat_file, root_dir, sigma = 0, train=True, transform=None):
        
        self.root_dir = root_dir
        self.transform = transform
        
        data = scio.loadmat(mat_file)
        images = data['images']
        score = images['score'][0][0][0]
        labels = images['labels'][0][0][0]
        RSD = np.transpose(images['RSD'][0][0])
        split = images['set'][0][0][0]
        meta = data['meta']
        imageList = meta['imageList'][0][0][0]
        
        train_list = imageList[split==1]
        test_list = imageList[split==3]
        
        train_distribution = RSD[split==1,:]
        test_distribution = RSD[split==3,:]
        
        train_labels = labels[split==1]
        test_labels = labels[split==3]
        
        train_score = score[split==1]
        test_score = score[split==3]
        
        index = (train_score >= 5 + sigma) | (train_score <= 5 - sigma)
        
        if train:
            self.image = train_list[index]
            self.distribution = train_distribution[index]
            self.labels = train_labels[index]
            self.score = train_score[index]
        else:
            self.image = test_list
            self.distribution = test_distribution
            self.labels = test_labels
            self.score = test_score
            
    def __len__(self):
        return len(self.score)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image[idx][0])
        image = pil_loader(img_name)

        label = self.labels[idx]
        distribution = self.distribution[idx,:]
        
        score = self.score[idx]

        sample = {'img_id': img_name, 'image': image, 'label': label, 'distribution': distribution, 'score': score}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample
    
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
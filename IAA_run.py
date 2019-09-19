#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 14:24:16 2018

@author: congbaobao
"""

import os

import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np


#from AVADataset import AVADataset
from ava_loader import AVADataset
from CUHKPQDataset import CUHKPQDataset
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

dim_h = 256
dim_mm = 4 * dim_h
R = 5
dropout = 0.5
activation = 'relu'

def weight_init(net): 
    for m in net.modules():    
        if isinstance(m, nn.Conv2d):         
            nn.init.kaiming_normal_(m.weight.data,nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data,nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class AbstractFusion(nn.Module):

    def __init__(self, opt={}):
        super(AbstractFusion, self).__init__()
        self.opt = opt

    def forward(self, input_v, input_q):
        raise NotImplementedError
        
        
    
class MLBFusion_pure(AbstractFusion):

    def __init__(self):
        super(MLBFusion_pure, self).__init__()
        
    def forward(self, input_v1, input_v2):
        
        batch_size = input_v1.size(0)
        channel = input_v1.size(1)
        x_v1 = input_v1
        x_v2 = input_v2
        # hadamard product
        x_fusion = torch.mul(x_v1, x_v2) 
        h = x_fusion.size(2)
        w = x_fusion.size(3)
        x_fusion = x_fusion.view(batch_size,channel,h*w)
        x_mm = x_fusion.sum(2)
        
        x_mm = torch.mul(torch.sign(x_mm),torch.sqrt(torch.abs(x_mm) + 1e-8))
        x_mm = torch.nn.functional.normalize(x_mm)
        
        return x_mm


class BaseCNN(nn.Module):

    def __init__(self, opt):
        nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.arch = 'resnet50'

        self.basemodel1 = torchvision.models.resnet50(pretrained=True)
        # load the pre-trained weights
        model_file = '%s_places365.pth.tar' % self.arch
        
        self.basemodel2 = torchvision.models.__dict__[self.arch](num_classes=365)
        #model_file = os.path.join('bilinear-cnn/src',model_file)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        self.basemodel2.load_state_dict(state_dict)

  
        self.map1_1 = nn.Sequential(nn.Conv2d(256,dim_h,1,bias = False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.map1_2 = nn.Sequential(nn.Conv2d(256,dim_h,1,bias = False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.map2_1 = nn.Sequential(nn.Conv2d(512,dim_h,1,bias = False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.map2_2 = nn.Sequential(nn.Conv2d(512,dim_h,1,bias = False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.map3_1 = nn.Sequential(nn.Conv2d(1024,dim_h,1,bias = False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.map3_2 = nn.Sequential(nn.Conv2d(1024,dim_h,1,bias = False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.map4_1 = nn.Sequential(nn.Conv2d(2048,dim_h,1,bias = False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.map4_2 = nn.Sequential(nn.Conv2d(2048,dim_h,1,bias = False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        
        weight_init(self.map1_1)
        weight_init(self.map1_2)  
        weight_init(self.map2_1)
        weight_init(self.map2_2)  
        weight_init(self.map3_1)
        weight_init(self.map3_2)    
        weight_init(self.map4_1)
        weight_init(self.map4_2)

        
        self.fusion = MLBFusion_pure()

        self.fc = torch.nn.Linear(dim_mm, 2)

        weight_init(self.fc)
        
        
        if opt['fc'] == True:
            # Freeze all previous layers.
            for param in self.basemodel1.parameters():
                param.requires_grad = False
            for param in self.basemodel2.parameters():
                param.requires_grad = False

        else:
            for param in self.basemodel1.parameters():
                param.requires_grad = True
            for param in self.basemodel2.parameters():
                param.requires_grad = True
                
                
            for param in self.basemodel1.conv1.parameters():
                param.requires_grad = False
            for param in self.basemodel1.bn1.parameters():
                param.requires_grad = False

            for param in self.basemodel2.conv1.parameters():
                param.requires_grad = False
            for param in self.basemodel2.bn1.parameters():
                param.requires_grad = False


    def forward(self, x):
        
        images = x
        
        x  = self.basemodel1.conv1(images)
        x = self.basemodel1.bn1(x)
        x = self.basemodel1.relu(x)
        x = self.basemodel1.maxpool(x)
        x1 = self.basemodel1.layer1(x)        
        x2 = self.basemodel1.layer2(x1)
        x3 = self.basemodel1.layer3(x2)        
        x4 = self.basemodel1.layer4(x3)
       
        y  = self.basemodel2.conv1(images)
        y = self.basemodel2.bn1(y)
        y = self.basemodel2.relu(y)
        y = self.basemodel2.maxpool(y)
        y1 = self.basemodel2.layer1(y)        
        y2 = self.basemodel2.layer2(y1)
        y3 = self.basemodel2.layer3(y2)        
        y4 = self.basemodel2.layer4(y3)
        

        
        x1_map = self.map1_1(x1) #14x14x1024 -> 14x14x1024
        x2_map = self.map2_1(x2)
        x3_map = self.map3_1(x3) #7x7x2048 -> 7x7x1024
        x4_map = self.map4_1(x4)
        
        y1_map = self.map1_2(y1) #14x14x1024 -> 14x14x1024
        y2_map = self.map2_2(y2)
        y3_map = self.map3_2(y3) #7x7x2048 -> 7x7x1024
        y4_map = self.map4_2(y4)


        fusion1 = self.fusion(x1_map, y1_map)
        fusion2 = self.fusion(x2_map, y2_map)
        fusion3 = self.fusion(x3_map, y3_map)
        fusion4 = self.fusion(x4_map, y4_map)
        
        fusion = torch.cat((fusion1, fusion2,fusion3,fusion4),1)      
        scores = self.fc(fusion).squeeze()
        
        return scores


class TrainManager(object):

    def __init__(self, options, path):
        print('Prepare the network and data.')
        self._options = options
        self._path = path      
        # Network.
        self._net = BaseCNN(options)
        if self._options['fc'] == True:
            self._solver = torch.optim.Adam(
                    self._net.parameters(), lr=self._options['base_lr'],
                    weight_decay=self._options['weight_decay'])
            self._scheduler = torch.optim.lr_scheduler.MultiStepLR(self._solver, milestones=[10], gamma=0.1)
        else:
            self._solver = torch.optim.Adam(
                    self._net.parameters(), lr=self._options['base_lr'],
                    weight_decay=self._options['weight_decay'])
            self._scheduler = torch.optim.lr_scheduler.MultiStepLR(self._solver, milestones=[20,40], gamma=0.1)
        self._net = nn.DataParallel(self._net, device_ids = [0, 1]).cuda()
        self._epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        if self._options['fc'] == False:
            self._net.load_state_dict(torch.load(os.path.join(self._path['fc_model'],'net_params_best.pkl')))
        
        
        self._net.train(True)

        print(self._net)

        self._criterion = nn.CrossEntropyLoss().cuda()

        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(size=386, scale=(0.5,1.0)), 
            torchvision.transforms.RandomCrop(336),
            torchvision.transforms.RandomHorizontalFlip(),      
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])

        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=386),  
            torchvision.transforms.CenterCrop(size=336), 
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
        if options['dataset'] == 'ava':
            train_data = AVADataset(
                    self._path['matfile'], self._path['ava'], sigma = 0, train=True,
                    transform=train_transforms)
            test_data = AVADataset(
                    self._path['matfile'], self._path['ava'], train=False,
                    transform=test_transforms)
        else:
            train_data = CUHKPQDataset(
                    self._path['train'], self._path['cuhkpq'],
                    transform=train_transforms)
            test_data = CUHKPQDataset(
                    self._path['val'], self._path['cuhkpq'],
                    transform=train_transforms)

        self._train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self._options['batch_size'],
            shuffle=True, num_workers=4, pin_memory=True)
        self._test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=self._options['batch_size'],
            shuffle=False, num_workers=4, pin_memory=True)
    def train(self):
        """Train the network."""
        print('Training.')
        best_acc = 0.0
        best_epoch = None
        print('Epoch\tTrain loss\tTrain acc\tTest acc')
        for t in range(self._epoch, self._options['epochs']):
            epoch_loss = []
            num_correct = 0.0
            num_total = 0.0
            iters = 0         
            self._scheduler.step()
            print(self._scheduler.get_lr())
            for data in tqdm(self._train_loader):
                X = data['image'].to(self.device)
                y = data['label'].to(self.device)

                # Clear the existing gradients.
                self._solver.zero_grad()
                # Forward pass.
                pred = self._net(X)
            
                loss = self._criterion(F.softmax(pred), y.detach())

                epoch_loss.append(loss.item())
                # Prediction.
                _, prediction = torch.max(pred.data, 1)
                num_total += y.size(0)
                num_correct += torch.sum(prediction == y.byte())
                # Backward pass.
                loss.backward()
                self._solver.step()
                iters = iters + 1

                
            train_acc = 100 * num_correct.item() / num_total
            with torch.no_grad():
                test_acc = self._accuracy(self._test_loader)
            
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = t + 1
                print('*', end='')
                
                pwd = os.getcwd()
                if self._options['fc'] == True:
                    modelpath = os.path.join(pwd,'fc_models','net_params_best.pkl')    
                else:
                    modelpath = os.path.join(pwd,'all_models',('net_params' + str(t) + '.pkl'))
                torch.save(self._net.state_dict(), modelpath)
                
                
            print('%d\t%4.3f\t\t%4.3f%%\t\t%4.3f%%' %
                  (t+1, sum(epoch_loss) / len(epoch_loss), train_acc, test_acc))
        print('Best at epoch %d, test accuaray %f' % (best_epoch, best_acc))

    def _accuracy(self, data_loader):

        self._net.train(False)
        num_correct = 0.0
        num_total = 0.0
        iters = 0
        for data in tqdm(data_loader):
            X = data['image'].to(self.device)
            y = data['label'].to(self.device)

            # Prediction.
            pred = self._net(X)
            _, prediction = torch.max(pred.data, 1)
            num_total += y.size(0)
            num_correct += torch.sum(prediction == y.byte())
            iters = iters + 1
            
        self._net.train(True)  # Set the model to training phase
        return 100 * num_correct.item() / num_total


def main(fc):
    """The main function."""
    import argparse
    parser = argparse.ArgumentParser(
        description='Train bilinear CNN on AVA.')
    parser.add_argument('--base_lr', dest='base_lr', type=float, required=True,
                        help='Base learning rate for training.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        required=True, help='Batch size.')
    parser.add_argument('--epochs', dest='epochs', type=int, required=True,
                        help='Epochs for training.')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                        required=True, help='Weight decay.')
    parser.add_argument('--dataset', dest='dataset', type=str,
                        required=True, help='dataset.')
#    parser.add_argument('--model', dest='model', type=str, required=True,
#                        help='Model for fine-tuning.')
    args = parser.parse_args()
    if args.base_lr <= 0:
        raise AttributeError('--base_lr parameter must >0.')
    if args.batch_size <= 0:
        raise AttributeError('--batch_size parameter must >0.')
    if args.epochs < 0:
        raise AttributeError('--epochs parameter must >=0.')
    if args.weight_decay <= 0:
        raise AttributeError('--weight_decay parameter must >0.')
    
    if fc == False:
        args.base_lr = 1e-6
        args.epochs = 50
    options = {
        'base_lr': args.base_lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
        'fc':fc,
        'dataset': args.dataset,   
        #'activation_mm':activation,
    }

    project_root = os.popen('pwd').read().strip()
    path = { 
        'ava': '/home/congbaobao/gloway/AVA_dataset/images/images',
        'cuhkpq': '/home/congbaobao/gloway/PhotoQualityDataset',
        'train': os.path.join(project_root,  'train.txt'),
        'val': os.path.join(project_root,  'val.txt'),
        'matfile': os.path.join(project_root,  'ava.mat'),
        'root':project_root,
        'fc_model': os.path.join(project_root,'fc_models'),
        'all_model': os.path.join(project_root,'all_models'),
    }


    manager = TrainManager(options, path)
    # manager.getStat()
    manager.train()


if __name__ == '__main__':
    fc = True
    main(fc)
    fc = False
    main(fc)

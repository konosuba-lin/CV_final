

import torch
from torch.utils.data.dataset import Dataset
import os
import numpy as np 

from torchvision.transforms import transforms
from PIL import Image
import pickle5
import random

from cfg import cfg

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.01):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_dataset_v0(root, ratio=0.9, cv=0):
    with open(root, 'rb') as f:
        data = pickle5.load(f)
        images, labels = data

    N = len(images)
    info = list(zip(images, labels))
    if(root==cfg['data_root']): random.shuffle(info)

    all_images, all_labels = zip(*info)

    x = int(N*ratio) 

    train_image = all_images[:x]
    val_image = all_images[x:]

    train_label = all_labels[:x] 
    val_label = all_labels[x:]

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ])
  
    # normally, we dont apply transform to test_set or val_set
    val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ])
    if(root==cfg['data_root']): #training
        prefix = './data/synthetics_train'
        train_set, val_set = cv_dataset(images=train_image, labels=train_label,transform=train_transform,prefix=prefix), \
                            cv_dataset(images=val_image, labels=val_label,transform=val_transform,prefix=prefix)
        return train_set, val_set
    else: #testing
        prefix = './data/aflw_val'
        test_set = cv_dataset(images=train_image, labels=train_label,transform=val_transform,prefix=prefix)
        return test_set

def get_dataset(root):
    with open(root+'annot.pkl', 'rb') as f:
        data = pickle5.load(f)
        images, labels = data
    N = len(images)
    info = list(zip(images, labels))
    if(root==cfg['data_root']): random.shuffle(info)
    images, labels = zip(*info)
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ])
    data_set = cv_dataset(images=images, labels=labels,transform=transform,prefix=root)
    return data_set
        

## TO DO ##
# Define your own cifar_10 dataset
class cv_dataset(Dataset):
    def __init__(self,images , labels=None , transform=None, prefix = './data/synthetics_train'):
        
        # It loads all the images' file name and correspoding labels here
        self.images = images 
        self.labels = labels 
        
        # The transform for the image
        self.transform = transform
        
        # prefix of the files' names
        self.prefix = prefix
        
        print(f'Number of images is {len(self.images)}')
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        ## TO DO ##
        # You should read the image according to the file path and apply transform to the images
        # Use "PIL.Image.open" to read image and apply transform
        path = os.path.join(self.prefix, self.images[idx])
        image = Image.open(path)
        image = self.transform(image)

        label = np.array(self.labels[idx])
        label = label.flatten()
        # You shall return image, label with type "long tensor" if it's training set
        return image, label
        

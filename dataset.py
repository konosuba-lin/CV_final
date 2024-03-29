

import torch
from torch.utils.data.dataset import Dataset
import os
import numpy as np 
from torchvision.transforms import transforms
import torchvision.transforms.functional as functional
from PIL import Image
import pickle
import random
import torchvision.transforms.functional as TF
from tool import random_rotate, random_flip, random_affine, rotate_img
from cfg import cfg


def get_dataset(root, start=0, end=1, aug=False, labeled=True, cvt=[False,0,False]):
    if labeled is True:
        with open(root+'annot.pkl', 'rb') as f:
            data = pickle.load(f)
            images, labels = data
        N = len(images)
        start,end = int(N*start),int(N*end)
        images, labels = images[start:end], labels[start:end]
        info = list(zip(images, labels))
        if(root==cfg['data_root']): random.shuffle(info)
        images, labels = zip(*info)
    else:
        images = [x for x in os.listdir(root) if x.endswith(".jpg")]
        labels = None

    if(root==cfg['data_root']):
        means = [0.3884782, 0.339793 , 0.3078828]
        stds  = [0.1900145 , 0.18243362, 0.18143885]
    else:
        means = [0.45668155, 0.38590077, 0.33730087]
        stds  = [0.27685005, 0.25711865, 0.24545431]

    if aug is True:
        transform = transforms.Compose([
                    transforms.ColorJitter(brightness=(0, 5), contrast=(0, 5), saturation=(0, 5), hue=(-0.1, 0.1)),
                    transforms.ToTensor(),
                    transforms.Normalize(means, stds),
                    transforms.RandomErasing(p=0.2),
                    ])
    else:
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(means, stds),
                    ])
    data_set = cv_dataset(images=images, labels=labels,transform=transform,prefix=root,aug=aug,cvt=cvt)
    return data_set


class cv_dataset(Dataset):
    def __init__(self,images , labels=None , transform=None, prefix = None, aug=False, cvt=[False,0,False]):
        self.images = images 
        self.labels = labels 
        self.transform = transform
        self.prefix = prefix
        self.aug = aug
        self.cvt = cvt
        print(f'Number of images is {len(self.images)}')
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        path = os.path.join(self.prefix, self.images[idx])
        image = Image.open(path)
        cvt_flag, angle, flip = self.cvt
        if self.labels is not None:
            label = np.array(self.labels[idx])
            if(self.aug):
                image, label = random_affine(image, label, use_random=True)
                image, label = random_rotate(image, label, use_random=True)
                image, label = random_flip(image, label, use_random=True)
            label = label.flatten()
        
        if cvt_flag:
            image = rotate_img(image,angle,flip)
        image = self.transform(image)
        if self.labels is not None:
            return image, label
        else:
            return image
        

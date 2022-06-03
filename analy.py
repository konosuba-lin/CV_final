import torch
import os
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim 
import torch.nn as nn
from tool import load_parameters
from MobileNet_v3 import MobileNetV3
from dataset import get_dataset
from tool import train, fixed_seed
from cfg import cfg
from PIL import Image
import pickle5
import random
from torchvision.transforms import transforms

def get_image_info(root):
    with open(root+'annot.pkl', 'rb') as f:
        img_paths, labels = pickle5.load(f)
    mean = [[],[],[]]
    std  = [[],[],[]]
    cnt = 0
    for i in img_paths:
        print(cnt,end="\r")
        cnt += 1
        path = os.path.join(root,i)
        img = Image.open(path)
        T = transforms.ToTensor()
        img = T(img)
        for i in range(3):
            mean[i].append(torch.mean(img[i]))
            std[i].append(torch.std(img[i]))
    mean = np.asarray(mean)
    std = np.asarray(std)
    return np.mean(mean,axis=1),np.mean(std,axis=1)

def main():
    model_path = cfg['model_path']
    data_root = cfg['data_root']
    eval_root = cfg['eval_root']
    batch_size = cfg['batch_size']
    seed = cfg['seed']
    # fixed random seed
    fixed_seed(seed)

    syn_info = get_image_info(root=data_root)
    print(syn_info)
    real_info = get_image_info(root=eval_root)
    print(real_info)

if __name__ == '__main__':
    main()
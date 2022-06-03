

from torchvision.transforms import transforms
from torch.utils.data import DataLoader

import torch 
import json
import argparse
from tqdm import tqdm

from tool import load_parameters, NME_loss, fixed_seed
from MobileNet_v3 import MobileNetV3
from dataset import get_dataset
from cfg import cfg
import numpy as np
import os
import cv2
from PIL import Image
import torchvision.transforms.functional as TF

# The function help you to calculate accuracy easily
# Normally you won't get annotation of test label. But for easily testing, we provide you this.
def draw(test_set,idx,model,device,save_path):
    path = os.path.join(test_set.prefix, test_set.images[idx])
    img = Image.open(path)
    label = np.array(test_set.labels[idx])
    origin_img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    for x, y in label:
        cv2.circle(origin_img, (int(x), int(y)), 2, (0, 0, 255), -1)
    label = label.flatten()
    label = torch.tensor(label).view(68,2)
    label = label.to(device)

    itr = 3
    for i in range(itr):
        data = test_set.transform(img)
        model.eval()
        with torch.no_grad():
            model.eval()
            data = data.to(device)
            output = model(data[None, ...]).view(68,2)
            for x, y in output:
                cv2.circle(origin_img, (int(x), int(y)), 2, (i*255//itr, 0, 0), -1)
            if(i==0): output_sum = output
            else: output_sum += output
        img = TF.adjust_gamma(img,gamma=0.3)
    output = output_sum/itr
    cv2.imwrite(save_path, origin_img)
    print(NME_loss(output,label))
def test_result(test_loader, model, device):
    pred = []
    nme = 0;
    model.eval()
    with torch.no_grad():
        model.eval()
        acc = 0 
        for batch_idx, ( data, label,) in enumerate(tqdm(test_loader)):
            data = data.to(device)
            label = label.to(device)
            output = model(data) 
            acc += (1 - NME_loss(output,label))

    return acc/(batch_idx+1)

def main():
    model_path = cfg['model_path']
    eval_root = cfg['eval_root']
    data_root = cfg['data_root']
    batch_size = cfg['batch_size']
    seed = cfg['seed']
    # fixed random seed
    fixed_seed(seed)

    ## TO DO ## 
    # Indicate the model you use here
    #model = myLeNet(num_out=10) 
    model = MobileNetV3(model_mode="LARGE", num_classes=136, multiplier=0.75)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    
    # Simply load parameters
    load_parameters(model=model, path=model_path)
    model.to(device)


    test_set = get_dataset(root=eval_root)
    #test_set = get_dataset(root=data_root,start=0,end=0.02,aug=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    acc = test_result(test_loader=test_loader, model=model, device=device)
    print("accuracy : ", acc)

    #print(img)
    idx = 163
    test_set = get_dataset(root=eval_root,aug=False) 
    draw(test_set,idx,model,device,save_path="test.png")

    '''MSE = []
    for i in range(199):
        MSE.append(draw(test_set,i,model,device,save_path="test.png"))
    MSE = np.array(MSE)
    print(max(MSE),np.argmax(MSE))'''

    
if __name__ == '__main__':
    main()
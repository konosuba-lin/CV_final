

from torchvision.transforms import transforms
from torch.utils.data import DataLoader

import torch 
import json
import argparse
from tqdm import tqdm

from tool import load_parameters, NME_loss
from MobileNet_v3 import MobileNetV3
from dataset import get_dataset
from cfg import cfg
import numpy as np
import os
import cv2
from PIL import Image

# The function help you to calculate accuracy easily
# Normally you won't get annotation of test label. But for easily testing, we provide you this.
def draw(test_set,idx,model,device):
    path = os.path.join(test_set.prefix, test_set.images[idx])
    image = Image.open(path)
    data = test_set.transform(image)
    label = np.array(test_set.labels[idx])
    T = transforms.ToPILImage()
    img = T(data)
    img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
    for x, y in label:
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
    label = label.flatten()
    label = torch.tensor(label)
    model.eval()
    with torch.no_grad():
        model.eval()
        data = data.to(device)
        label = label.to(device)
        output = model(data[None, ...])
        output = output.view(68,2)
    for x, y in output:
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
    cv2.imwrite("test.png", img)
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
    batch_size = cfg['batch_size']
    # change your model here 

    ## TO DO ## 
    # Indicate the model you use here
    #model = myLeNet(num_out=10) 
    model = MobileNetV3(model_mode="SMALL", num_classes=136, multiplier=1.0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    
    # Simply load parameters
    load_parameters(model=model, path=model_path)
    model.to(device)


    test_set = get_dataset(root=eval_root)    
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    acc = test_result(test_loader=test_loader, model=model, device=device)
    print("accuracy : ", acc)

    #print(img)
    draw(test_set,2,model,device)

    
if __name__ == '__main__':
    main()
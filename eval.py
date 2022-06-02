

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

import random

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
        cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)
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
        cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)
    cv2.imwrite(test_set.images[idx].replace('image', 'test'), img)
    print(test_set.images[idx], "NME: ", NME_loss(output,label)*100)

def validation_test(data_loader, model, device):
    nme = 0;
    model.eval()
    with torch.no_grad():
        model.eval()
        acc = 0 
        for batch_idx, ( data, label,) in enumerate(tqdm(data_loader)):
            data = data.to(device)
            label = label.to(device)
            output = model(data) 
            acc += (1 - NME_loss(output,label))

    return acc/(batch_idx+1)

def generate_result(dataset, data_loader, model, device):
    pred = []
    model.eval()
    with torch.no_grad():
        model.eval()
        for batch_idx, data in enumerate(tqdm(data_loader)):
            data = data.to(device)
            output = model(data) 
            pred.append(output)
    # print(dataset.images[0])
    # print(pred[0][0].view(68,2))
    f = open('solution.txt', 'w')
    idx = 0
    for batch in pred:
        batch.cpu().detach().numpy().tolist()
        for result in batch:
            f.writelines([dataset.images[idx], ' '])
            lines = [str(x.cpu().detach().numpy()) for x in result]
            lines = list(' '.join(lines))
            f.writelines(lines)
            f.writelines('\n')
            idx = idx + 1
    f.close()
    

def main():
    model_path = cfg['model_path']
    eval_root = cfg['eval_root']
    test_root = cfg['test_root']
    batch_size = cfg['batch_size']
    # change your model here 

    ## TO DO ## 
    # Indicate the model you use here
    #model = myLeNet(num_out=10) 
    model = MobileNetV3(model_mode="LARGE", num_classes=136, multiplier=0.5)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    
    # Simply load parameters
    load_parameters(model=model, path=model_path)
    model.to(device)

    val_set = get_dataset(root=eval_root)    
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    acc = validation_test(data_loader=val_loader, model=model, device=device)
    print("validation NME: ", (1 - acc)*100)
    
    ##### choose your image number to be test #####
    # you can choose the test image you want
    test_image = random.randint(0, len(val_set)-1)
    # test_image = 2
    print("\n######## testing random sample... ########")
    draw(val_set, test_image, model, device)

    ##### generate solution #####
    print("\n######## generating solution... ########")
    test_set = get_dataset(root=test_root, labeled=False)    
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    generate_result(dataset=test_set, data_loader=test_loader, model=model, device=device)

    print("\n!!!!! Note: The output is solution.txt and submission.zip !!!!!")

    
if __name__ == '__main__':
    main()
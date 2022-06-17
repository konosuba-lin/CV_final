

from torchvision.transforms import transforms
from torch.utils.data import DataLoader

import torch 
import json
import argparse
from tqdm import tqdm

from tool import load_parameters, NME_loss, fixed_seed, random_rotate, rotate_img, rotate_label
from dataset import get_dataset
from cfg import cfg
import numpy as np
import os
import cv2
from PIL import Image
import torchvision.transforms.functional as TF
from ShuffleNet import shufflenetv2
from matplotlib import pyplot as plt
import random
from scipy import stats
# The function help you to calculate accuracy easily
# Normally you won't get annotation of test label. But for easily testing, we provide you this.
def draw(save_path,label,output,img):
    cv2_img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    for x, y in label:
        cv2.circle(cv2_img, (int(x), int(y)), 2, (0, 0, 255), -1)
    for x, y in output:
        cv2.circle(cv2_img, (int(x), int(y)), 2, (0, 255, 0), -1)
    cv2.imwrite(save_path,cv2_img)

def box_area(lab):
    l,r = min(lab[:,0]), max(lab[:,0])
    t,b = min(lab[:,1]), max(lab[:,1])
    area = (r-l)*(b-t)
    return area

def yaw(lab):
    pl1,pl2,pr1,pr2 = lab[39],lab[36],lab[42],lab[45]
    r1 = np.linalg.norm(pl1-pl2,ord=2)
    r2 = np.linalg.norm(pr1-pr2,ord=2)
    r = r1/r2-r2/r1
    #r = r1/r2
    return r

def pitch(lab):
    p1 = (lab[2]+lab[14])/2
    p2 = lab[33]
    p = p1-p2
    theta = np.arctan(p[1]/abs(p[0]))*(180/np.pi)
    return theta

def roll(lab):
    p = lab[8]-lab[33]
    theta = np.arctan(p[0]/abs(p[1]))
    theta = theta*(180/np.pi)
    return theta

def loss(lab):
    lab2 = np.zeros_like(lab)
    lab2[:] = lab[:]
    idx = [17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1] +\
    [27,26,25,24,23,22,21,20,19,18] +\
    [28,29,30,31] +\
    [36,35,34,33,32] +\
    [46,45,44,43,48,47] +\
    [40,39,38,37,42,41] +\
    [55,54,53,52,51,50,49] +\
    [60,59,58,57,56] +\
    [65,64,63,62,61,68,67,66]
    idx = np.array(idx)-1
    lab2 = lab2[idx]
    lab2 = (lab+lab2)/2
    res = stats.linregress(x=lab2[17:,0], y=lab2[17:,1])
    loss = (res.rvalue)**2
    return loss,lab2[17:]


def validate_result(eval_loader, model, device, cvt):
    nme = 0;
    model.eval()
    outputs = []
    labels = []
    cvt_flag, angle, flip = cvt
    with torch.no_grad():
        model.eval()
        for batch_idx, ( data, label,) in enumerate(tqdm(eval_loader)):
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            output = output.view(-1,68,2).cpu().detach().numpy()
            label = label.view(-1,68,2).cpu().detach().numpy()
            for i in range(len(output)):
                if cvt_flag:
                    output[i] = rotate_label(output[i],-angle,flip,dir="backward")
                outputs.append(output[i])
                labels.append(label[i])
            nme += NME_loss((torch.tensor(output)).to(device),(torch.tensor(label)).to(device))
    return nme/(batch_idx+1),np.array(outputs),np.array(labels)

def validating(model,device):
    print("------------START-------------")
    eval_root = cfg['eval_root']
    batch_size = 16
    cvt_cfgs = [[False,0,False],[True,0,True]]
    outs = []
    labs = []
    for cvt_cfg in cvt_cfgs:
        eval_set = get_dataset(root=eval_root,cvt=cvt_cfg)
        eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)
        nme,out,lab = validate_result(eval_loader=eval_loader, model=model, device=device, cvt=cvt_cfg)
        outs.append(out)
        labs.append(lab)
        print("cvt_flag: %d, angle: %2d, flip: %d, NME: %.4f "%(cvt_cfg[0],cvt_cfg[1],cvt_cfg[2],nme*100))

    out = []
    lab = labs[0] #labs are all equal
    infos = []
    nmes = []
    random.seed(0)
    weights = np.array([0.25,0.25,0.25,0.25])
    for i in range(len(outs[0])):
        candidates = np.array([outs[j][i] for j in range(len(outs))])
        best_out = np.mean(candidates,axis=0)
        nme = NME_loss((torch.tensor(best_out)).to(device),(torch.tensor(lab[i])).to(device))
        nmes.append(nme)
        out.append(best_out)
        infos.append([nme*100]+[i])
        
    nmes = np.array(nmes)
    infos = np.array(infos)
    nme= np.mean(nmes)
    infos = sorted(infos,key=lambda x:x[0],reverse=False)
    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)
    for i in infos[-4:]:
        print(i)
    print("-------Validation Result--------")
    print("NME: %.8f"%(nme*100))

def test_result(test_loader, model, device, cvt):
    model.eval()
    outputs = []
    cvt_flag, angle, flip = cvt
    with torch.no_grad():
        model.eval()
        for batch_idx, data in enumerate(tqdm(test_loader)):
            data = data.to(device)
            output = model(data)
            output = output.view(-1,68,2).cpu().detach().numpy()
            for i in range(len(output)):
                if cvt_flag:
                    output[i] = rotate_label(output[i],-angle,flip,dir="backward")
                outputs.append(output[i])
    return np.array(outputs)

def testing(model,device):
    print("------------START-------------")
    test_root = cfg['test_root']
    batch_size = 16
    cvt_cfgs = [[False,0,False],[True,0,True]]
    outs = []
    for cvt_cfg in cvt_cfgs:
        test_set = get_dataset(root=test_root,cvt=cvt_cfg,labeled=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        out = test_result(test_loader=test_loader, model=model, device=device, cvt=cvt_cfg)
        outs.append(out)
    out = np.median(outs,axis=0)
    f = open('solution.txt', 'w')
    for idx in range(len(out)):
        f.writelines([test_set.images[idx], ' '])
        lines = [str(x[0])+' '+str(x[1]) for x in out[idx]]
        lines = list(' '.join(lines))
        f.writelines(lines)
        f.writelines('\n')
    f.close()
    print("---Finish Generating Result----")


def main():
    model_path = cfg['model_path']
    seed = cfg['seed']
    # fixed random seed
    fixed_seed(369)

    #model = MobileNetV3(model_mode="LARGE", num_classes=136, multiplier=0.75)
    model = shufflenetv2(num_classes=136)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    load_parameters(model=model, path=model_path)
    model.to(device)

    #validating(model,device)
    #affine_test(model,device,0)
    testing(model,device)
    
if __name__ == '__main__':
    main()

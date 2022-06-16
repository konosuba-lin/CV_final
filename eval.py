

from torchvision.transforms import transforms
from torch.utils.data import DataLoader

import torch 
import json
import argparse
from tqdm import tqdm

from tool import load_parameters, NME_loss, fixed_seed, random_rotate, random_flip, find_angle_from_label
from MobileNet_v3 import MobileNetV3
from ShuffleNet import shufflenetv2
from dataset import get_dataset
from cfg import cfg
import numpy as np
import os
import cv2
from PIL import Image
import torchvision.transforms.functional as TF

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

def postprocess(test_set, idx, model, device, save_path="test.png", draw=True, is_eval=True):
    path = os.path.join(test_set.prefix, test_set.images[idx])
    img = Image.open(path)

    if is_eval is True:
        label = np.array(test_set.labels[idx])
        origin_img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        if draw is True:
            # cv2.circle(origin_img, (int(label[39][0]), int(label[39][1])), 2, (0, 0, 255), -1) # left eye
            # cv2.circle(origin_img, (int(label[42][0]), int(label[42][1])), 2, (0, 0, 255), -1) # right eye
            # cv2.circle(origin_img, (int(label[33][0]), int(label[33][1])), 2, (0, 0, 255), -1) # nose
            # cv2.circle(origin_img, (int(label[28][0]), int(label[28][1])), 2, (0, 0, 255), -1) # between eyes
            for x, y in label:
                cv2.circle(origin_img, (int(x), int(y)), 2, (0, 0, 255), -1)

    data = test_set.transform(img)
    model.eval()
    with torch.no_grad():
        model.eval()
        data = data.to(device)
        pred_label = model(data[None, ...]).view(68,2)

    pred_label = pred_label.cpu().numpy()

    flipped_img, _ = random_flip(img=img, use_random=False, flip=True)
    data = test_set.transform(flipped_img)
    model.eval()
    with torch.no_grad():
        model.eval()
        data = data.to(device)
        flipped_label = model(data[None, ...]).view(68,2)
    
    flipped_label = flipped_label.cpu().numpy()

    _, flipped_label = random_flip(img=flipped_img, label=flipped_label, use_random=False, flip=True)

    # output = pred_label
    output = (flipped_label + pred_label)/2
    # output = np.median(flipped_label, pred_label)

    # pred_list.append(pred_label.cpu().numpy())
    
    # output = np.mean(pred_list, axis=0)

    # rot_list = [0, -2,  2, -4, 4, -6, 6]
    # pred_list = []
    # for rot in rot_list:
    #     rot_img, _ = random_rotate(img=img, use_random=False, angle=rot)
    #     data = test_set.transform(rot_img)
    #     model.eval()
    #     with torch.no_grad():
    #         model.eval()
    #         data = data.to(device)
    #         pred_label = model(data[None, ...]).view(68,2)
    #     pred_list.append(pred_label.cpu().numpy())

    # print(output)
    
    # angle = find_angle_from_label(pred_label.cpu())
    # print(angle)
    # max_iter = 10
    # if abs(angle > 10):
    # if False:
    #     for i in range(max_iter):
    #         img, _ = random_rotate(img=img, use_random=False, angle=-angle)
    #         data = test_set.transform(img)
    #         with torch.no_grad():
    #             model.eval()
    #             data = data.to(device)
    #             new_pred_label = model(data[None, ...]).view(68,2)
            
    #         new_angle = find_angle_from_label(new_pred_label.cpu())
    #         print(new_angle)

    #         if abs(new_angle - angle) < 3:
    #             break
    #         else:
    #             angle = new_angle
    #     output = new_pred_label
    # else: 
    #     output = pred_label

    
    if is_eval is True:
        output = torch.tensor(output).to(device)
        if draw is True:
            for x, y in output:
                cv2.circle(origin_img, (int(x), int(y)), 2, (255, 0, 0), -1)
            cv2.imwrite(save_path, origin_img)

        label = torch.tensor(label).to(device)
        return NME_loss(output, label)*100
    else:
        return output

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

def generate_result_with_postprocess(dataset, data_loader, model, device):

    f = open('solution.txt', 'w')

    print(len(dataset))
    for idx in range(len(dataset)):
        output = postprocess(dataset, idx, model, device, is_eval=False)
        output = output.flatten()
        f.writelines([dataset.images[idx], ' '])
        lines = [str(x) for x in output]
        lines = list(' '.join(lines))
        f.writelines(lines)
        f.writelines('\n')
        idx = idx + 1

    f.close()
    

def main():
    model_path = cfg['model_path']
    eval_root = './data/aflw_val/'
    test_root = cfg['test_root']
    batch_size = cfg['batch_size']
    num_out = cfg['num_out']
    seed = cfg['seed']
    # fixed random seed
    fixed_seed(seed)

    # Indicate the model you use here
    # model = MobileNetV3(model_mode="LARGE", num_classes=num_out, multiplier=0.75)
    model = shufflenetv2(num_classes=num_out)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    
    # Simply load parameters
    load_parameters(model=model, path=model_path)
    model.to(device)

    val_set = get_dataset(root=eval_root)    
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    print("\n######## validating... ########")

    idx = 44
    testNME = postprocess(val_set, idx, model, device, save_path="test.jpg")
    print("TEST NME: ", testNME)
    
    LOSS = []
    for i in range(len(val_set)):
        LOSS.append(postprocess(val_set, i, model, device, draw=False, is_eval=True))
    LOSS = np.array(LOSS)

    print(LOSS.shape)
    print("MAX LOSS =", max(LOSS), "at image", np.argmax(LOSS))
    print("Total validation NME: ", sum(LOSS)/len(val_set))
    
    acc = validation_test(data_loader=val_loader, model=model, device=device)
    print("Total validation NME: ", (1 - acc)*100)
    
    # ##### choose your image number to be test #####
    # # you can choose the test image you want
    # test_image = random.randint(0, len(val_set)-1)
    # # test_image = 2
    # print("\n######## testing random sample... ########")
    # draw(val_set, test_image, model, device)

    ##### generate solution #####
    print("\n######## generating solution... ########")
    test_set = get_dataset(root=test_root, labeled=False)    
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    generate_result_with_postprocess(dataset=test_set, data_loader=test_loader, model=model, device=device)

    print("\n!!!!! Note: The output is solution.txt and submission.zip !!!!!")

    
if __name__ == '__main__':
    main()
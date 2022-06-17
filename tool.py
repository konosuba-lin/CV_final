import torch
import torch.nn as nn

import numpy as np 
import time
from tqdm import tqdm
import os
import random
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from cfg import cfg
from scipy.ndimage.interpolation import rotate

def fixed_seed(myseed):
    np.random.seed(myseed)
    random.seed(myseed)
    torch.manual_seed(myseed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        torch.cuda.manual_seed(myseed)
    

def save_model(model, path):
    print(f'Saving model to {path}...')
    torch.save(model.state_dict(), path)
    print("End of saving !!!")


def load_parameters(model, path):
    print(f'Loading model parameters from {path}...')
    param = torch.load(path, map_location='cuda')
    model.load_state_dict(param)
    print("End of loading !!!")



## TO DO ##
def plot_learning_curve(x, y, name):
    plt.figure()
    plt.plot(x, y)
    plt.title(name)
    plt.savefig("{}.png".format(name))

    pass

def random_rotate(img, label=[], use_random=True, angle=0):
    max_rot = 30

    if use_random is True:
        angle = random.randint(-max_rot, max_rot)
    else:
        angle = angle
    
    img = TF.rotate(img, angle) # counter clockwise
    angle = -angle/180*np.pi # since y axis toward negative
    M = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    C = np.array([384/2,384/2])
    for i in range(len(label)):
        label[i] = M.dot(label[i]-C)+C
    return img, label

def random_affine(img, label=[], use_random=True, translate=[0,0], scale=1):
    max_translate = 10
    max_scale = 1.15

    if use_random is True:
        translate = [random.randint(-max_translate, max_translate),random.randint(-max_translate, max_translate)]
        scale = random.uniform(1/max_scale, max_scale)
    else:
        translate = translate
        scale = scale
    
    img = TF.affine(img=img,angle=0,shear=0,translate=translate,scale=scale)
    c0 = np.array([192,192])
    c1 = c0 + np.array(translate)
    label = (label-c0)*scale+c1
    return img, label
    
def random_flip(img, label=np.zeros((68,2)), use_random=True, flip=True):
    if use_random is True:
        flip = bool(random.getrandbits(1))
    else:
        flip = flip

    output = np.zeros_like(label)
    output[:,:] = label[:,:]

    if flip is True:
        img = TF.hflip(img)

        output[:,0] = 384 - output[:,0]
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
        output = output[idx]

    return img, output

def rotate_img(img,angle,flip):
    #first flip then rotate
    img2 = img
    if(flip):
        img2 = TF.hflip(img2)
    img2 = TF.rotate(img2, angle) #counter clockwise
    return img2

def rotate_label(label,angle,flip,dir):
    if(dir=="forward"): #first flip then rotate
        label2 = np.zeros_like(label)
        label2[:,:] = label[:,:]
        if(flip):
            label2[:,0] = 384 - label2[:,0]
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
            label2 = label2[idx]
        angle = -angle/180*np.pi # since y axis toward negative
        M = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        C = np.array([384/2,384/2])
        for i in range(len(label2)):
            label2[i] = M.dot(label2[i]-C)+C
    else: #first rotate then flip
        label2 = np.zeros_like(label)
        angle = -angle/180*np.pi # since y axis toward negative
        M = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        C = np.array([384/2,384/2])
        for i in range(len(label2)):
            label2[i] = M.dot(label[i]-C)+C
        if(flip):
            label2[:,0] = 384 - label2[:,0]
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
            label2 = label2[idx]
    return label2


def find_angle_from_label(label):
    between_eyes = label[28]
    nose = label[33]
    left_eye = label[39]
    right_eye = label[42]
    
    # find eye mean
    eye_mean = (left_eye + right_eye)/2
    # find regression line of 3 points
    points = np.vstack((eye_mean, between_eyes, nose))
    slope, _ = np.polyfit(points[:,0], points[:,1], 1)

    # return angle of the line
    angle = np.arctan(slope)
    if angle > 0:
        angle = angle - np.pi/2
    else:
        angle = angle + np.pi/2
    return angle*360/(2*np.pi)

def NME_loss(output,label):
    label = label.view(-1,68,2)
    output = output.view(-1,68,2)
    loss = (output-label)
    loss = loss.cpu().detach().numpy()
    loss = np.mean(np.sqrt(np.sum(np.power(loss,2),axis=2))/384)
    return loss

def train(model, train_loader, val_loader, num_epoch, log_path, save_path, device, criterion, scheduler, optimizer):
    start_train = time.time()

    overall_loss = np.zeros(num_epoch ,dtype=np.float32)
    overall_acc = np.zeros(num_epoch ,dtype = np.float32)
    overall_val_loss = np.zeros(num_epoch ,dtype=np.float32)
    overall_val_acc = np.zeros(num_epoch ,dtype = np.float32)

    best_acc = 0
    for i in range(num_epoch):
        print(f'epoch = {i}')
        start_time = time.time()
        train_loss = 0.0 
        corr_num = 0

        model.train()
        for batch_idx, ( data, label,) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            label = label.to(device)
            output = model(data) 
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm= 5.)
            optimizer.step()
            train_loss += loss.item()
            # print("NME = {:.2f}".format(NME_loss(output,label)*100))
            corr_num += (1 - NME_loss(output,label))
        scheduler.step()
        train_loss = train_loss / (batch_idx+1)
        train_acc = corr_num / (batch_idx+1)
        overall_loss[i], overall_acc[i] = train_loss, train_acc
        with torch.no_grad():
            model.eval()
            val_loss = 0
            corr_num = 0
            val_acc = 0 
            model.eval()
            
            for batch_idx, ( data, label,) in enumerate(tqdm(val_loader)):
                data = data.to(device)
                label = label.to(device)
                output = model(data) 
                loss = criterion(output, label)
                val_loss += loss.item()
                corr_num += (1 - NME_loss(output,label))
            val_loss = val_loss / (batch_idx+1)
            val_acc = corr_num / (batch_idx+1)
            overall_val_loss[i], overall_val_acc[i] = val_loss, val_acc

        # Display the results
        end_time = time.time()
        elp_time = end_time - start_time
        min = elp_time // 60 
        sec = elp_time % 60
        print('*'*10)
        print('time = {:.4f} MIN {:.4f} SEC, total time = {:.4f} Min {:.4f} SEC '.format(elp_time // 60, elp_time % 60, (end_time-start_train) // 60, (end_time-start_train) % 60))
        print(f'training loss : {train_loss:.4f} ', f' train NME = {(1-train_acc)*100:2.2f}' )
        print(f'val loss : {val_loss:.4f} ', f' val NME = {(1-val_acc)*100:2.2f}' )
        print('========================\n')

        with open(log_path, 'a') as f :
            f.write(f'epoch = {i}\n', )
            f.write('time = {:.4f} MIN {:.4f} SEC, total time = {:.4f} Min {:.4f} SEC\n'.format(elp_time // 60, elp_time % 60, (end_time-start_train) // 60, (end_time-start_train) % 60))
            f.write(f'training loss : {train_loss}  train acc = {train_acc}\n' )
            f.write(f'val loss : {val_loss}  val acc = {val_acc}\n' )
            f.write('============================\n')

        if  val_acc > best_acc:
            best_acc = val_acc
            
        if  i == num_epoch-1: 
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pt'))


    x = range(0,num_epoch)
    overall_acc = overall_acc.tolist()
    overall_loss = overall_loss.tolist()
    overall_val_acc = overall_val_acc.tolist()
    overall_val_loss = overall_val_loss.tolist()

    plot_learning_curve(x, overall_acc, "overall_acc")
    plot_learning_curve(x, overall_loss, "overall_loss")
    plot_learning_curve(x, overall_val_acc, "overall_val_acc")
    plot_learning_curve(x, overall_val_loss, "overall_val_loss")
    
    pass
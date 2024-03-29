import torch
import os
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim 
import torch.nn as nn

from tool import load_parameters

from ShuffleNet import shufflenetv2
from dataset import get_dataset
from tool import train, fixed_seed
from cfg import cfg
from torchsummary import summary

class NMELoss(nn.Module):
    def __init__(self):
        # --------------------------------------------
        # Initialization
        # --------------------------------------------
        super(NMELoss, self).__init__()

    def forward(self, outputs, labels):
        # --------------------------------------------
        # Define forward pass
        # --------------------------------------------
        labels = labels.view(-1,68,2)
        outputs = outputs.view(-1,68,2)
        loss = torch.sum(torch.sqrt(torch.sum(torch.pow(outputs-labels,2),dim=2).to(torch.double)))
        return loss
        


def train_interface():
    data_root = cfg['data_root']
    eval_root = cfg['eval_root']
    model_type = cfg['model_type']
    num_out = cfg['num_out']
    num_epoch = cfg['num_epoch']
    seed = cfg['seed']
    model_path = cfg['model_path']
    
    fixed_seed(seed)
    
    os.makedirs( os.path.join('./acc_log',  model_type), exist_ok=True)
    os.makedirs( os.path.join('./save_dir', model_type), exist_ok=True)    
    log_path = os.path.join('./acc_log', model_type, 'acc_' + model_type + '_.log')
    save_path = os.path.join('./save_dir', model_type)
    with open(log_path, 'w'):
        pass
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Cuda = ", end = '')
    print(torch.cuda.is_available())
    
    lr = cfg['lr']
    batch_size = cfg['batch_size']
    milestones = cfg['milestones']

    ## MODEL DECLARATION ##
    model = shufflenetv2(num_classes=num_out)

    summary(model.cuda(), input_size=(3, 384, 384))

    train_set = get_dataset(root=data_root,aug=True)
    val_set   = get_dataset(root=eval_root)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=milestones, gamma=0.1)
    criterion = NMELoss()

    model = model.to(device)
    train(model=model,train_loader=train_loader, val_loader=val_loader, 
          num_epoch=num_epoch, log_path=log_path, save_path=save_path,
          device=device, criterion=criterion, optimizer=optimizer, scheduler=scheduler)

    '''train_set = get_dataset(root=data_root,aug=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    model = MobileNetV3(model_mode="LARGE", num_classes=136, multiplier=0.75)
    load_parameters(model=model, path=model_path)
    model.to(device)
    for name, param in model.named_parameters():
            if not 'out_conv2' in name:
                param.requires_grad = False
    train(model=model,train_loader=train_loader, val_loader=val_loader, 
          num_epoch=num_epoch, log_path=log_path, save_path=save_path,
          device=device, criterion=criterion, optimizer=optimizer, scheduler=scheduler)'''

if __name__ == '__main__':
    train_interface()
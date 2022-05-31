
import torch
import os
import numpy as np


from torch.utils.data import DataLoader
import torch.optim as optim 
import torch.nn as nn
# from torchsummary import summary

from model import myLeNet
from dataset import get_dataset
from tool import train, fixed_seed

from cfg import cfg
class NMELoss(nn.Module):
    def __init__(self):
        # --------------------------------------------
        # Initialization
        # --------------------------------------------
        super(MSELoss, self).__init__()

    def forward(self, outputs, labels):
        # --------------------------------------------
        # Define forward pass
        # --------------------------------------------
        labels = labels.view(-1,2)
        outputs = outputs.view(-1,2)
        loss = torch.mean(torch.sum(torch.sqrt(torch.sum(torch.pow(outputs-labels,2),dim=1).to(torch.double))))/384
        return loss
        


def train_interface():
    
    """ input argumnet """

    data_root = cfg['data_root']
    model_type = cfg['model_type']
    num_out = cfg['num_out']
    num_epoch = cfg['num_epoch']
    split_ratio = cfg['split_ratio']
    seed = cfg['seed']
    
    # fixed random seed
    fixed_seed(seed)
    

    os.makedirs( os.path.join('./acc_log',  model_type), exist_ok=True)
    os.makedirs( os.path.join('./save_dir', model_type), exist_ok=True)    
    log_path = os.path.join('./acc_log', model_type, 'acc_' + model_type + '_.log')
    save_path = os.path.join('./save_dir', model_type)


    with open(log_path, 'w'):
        pass
    
    ## training setting ##
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu') 
    print("Cuda = ", end = '')
    print(torch.cuda.is_available())
    
    
    """ training hyperparameter """
    lr = cfg['lr']
    batch_size = cfg['batch_size']
    milestones = cfg['milestones']
    
    
    ## Modify here if you want to change your model ##
    model = myLeNet(num_out=num_out)
    # print model's architecture
    # print(model)

    # Get your training Data 
    ## TO DO ##
    # You need to define your cifar10_dataset yourself to get images and labels for earch data
    # Check myDatasets.py 
      
    train_set, val_set = get_dataset(root=data_root, ratio=split_ratio)    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    # define your loss function and optimizer to unpdate the model's parameters.
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=milestones, gamma=0.2)
    
    # We often apply crossentropyloss for classification problem. Check it on pytorch if interested
    criterion = nn.MSELoss()
    #criterion = NMELoss()
    
    # Put model's parameters on your device
    model = model.to(device)
    # summary(model, input_size=(3,32,32), batch_size=batch_size)

    ### TO DO ### 
    # Complete the function train
    # Check tool.py
    train(model=model, train_loader=train_loader, val_loader=val_loader, 
          num_epoch=num_epoch, log_path=log_path, save_path=save_path,
          device=device, criterion=criterion, optimizer=optimizer, scheduler=scheduler)

if __name__ == '__main__':
    train_interface()
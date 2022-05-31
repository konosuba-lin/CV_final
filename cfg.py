cfg = {
    'model_type': 'LeNet',
    'data_root' : './data/synthetics_train/annot.pkl',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 369,
    
    # training hyperparameters
    'batch_size': 64,
    'lr':0.01,
    'milestones': [15, 20, 25],
    'num_out': 68*2,
    'num_epoch': 30,
    
}
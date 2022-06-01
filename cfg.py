cfg = {
    'model_type': 'LeNet',
    'data_root' : './data/synthetics_train/',
    'eval_root' : './data/aflw_val/',
    'model_path': './save_dir/LeNet/best_model.pt',
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 369,
    
    # training hyperparameters
    'batch_size': 64,
    'lr':0.01,
    'milestones': [3, 6, 9, 12, 15, 18],
    'num_out': 68*2,
    'num_epoch': 30,

}
cfg = {
    'model_type': 'ShuffleNet',
    'data_root' : './data/synthetics_train/',
    'eval_root' : './data/aflw_val/',
    'test_root' : './data/aflw_test/',
    'model_path': './save_dir/ShuffleNet/best_model.pt',
    # 'model_path': './save_dir/ShuffleNet/best_model_0.75_aug.pt',
    # ratio of training images and validation images 
    # 'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 222,
    
    # training hyperparameters
    'batch_size': 32,
    'lr':0.01,
    'milestones': [5,10,15,20,25],
    'num_out': 68*2,
    'num_epoch': 15,
}

# cfg = {
#     'model_type': 'MobileNet',
#     'data_root' : './data/synthetics_train/',
#     'eval_root' : './data/aflw_val/',
#     'test_root' : './data/aflw_test/',
#     'model_path': './save_dir/MobileNet/best_model.pt',
#     # ratio of training images and validation images 
#     #'split_ratio': 0.9,
#     # set a random seed to get a fixed initialization 
#     'seed': 369,
    
#     # training hyperparameters
#     'batch_size': 32,
#     'lr':0.001,
#     'milestones': [5,10,15],
#     'num_out': 68*2,
#     'num_epoch': 20,
# }

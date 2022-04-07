import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
# from tqdm.notebook import tqdm

module_path = os.path.abspath(os.path.join('../../src/'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import ResNet as net

#===========================================================================================================

# adjustables

k = 5                       # model index: should be in {0, 2, ..., 10}
dt = 0.01                     # time unit: 0.0005 for Lorenz and 0.01 for others
system = 'VanDerPol'         # system name: 'Hyperbolic', 'Cubic', 'VanDerPol', 'Hopf' or 'Lorenz'
noise = 0.0                   # noise percentage: 0.00, 0.01 or 0.02

lr = 1e-3                     # learning rate
max_epoch = 50000            # the maximum training epoch 
batch_size = 320              # training batch size
arch = [2, 512, 512, 512, 2]  # architecture of the neural network

# paths
data_dir = os.path.join('../../data/', system)
model_dir = os.path.join('../../models/', system)

# global const
n_forward = 5
step_size = 2**k

percent_data_list = [ 0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625]

        
# load data
train_data = np.load(os.path.join(data_dir, 'train_noise{}.npy'.format(noise)))
val_data = np.load(os.path.join(data_dir, 'val_noise{}.npy'.format(noise)))
test_data = np.load(os.path.join(data_dir, 'test_noise{}.npy'.format(noise)))
n_train = train_data.shape[0]
n_val = val_data.shape[0]
n_test = test_data.shape[0]

print("train_data shape = ", train_data.shape)
print("val_data shape = ", val_data.shape)
print("test_data.shape = ", test_data.shape)

for percent_data in percent_data_list:

    # create dataset object
    dataset = net.DataSet(train_data[:int(n_train*percent_data)], val_data, test_data, dt, step_size, n_forward)

    #make and train
    model_name = 'model_D{}_noise{}_data{}.pt'.format(step_size, noise, percent_data)

    # # create/load model object
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = torch.load(os.path.join(model_dir, model_name), map_location=device)
        model.device = device
        print("model loaded ", model_name)
    except:
      

        # create/load model object
        print('create model {} ...'.format(model_name))
        model = net.ResNet(arch=arch, dt=dt, step_size=step_size)#, prev_models=prev_models)

        # training
        model.train_net(dataset, max_epoch=max_epoch, batch_size=batch_size, lr=lr,
                        model_path=os.path.join(model_dir, model_name))
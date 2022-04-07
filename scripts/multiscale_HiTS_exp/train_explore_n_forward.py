import os
import sys
import torch
import numpy as np

module_path = os.path.abspath(os.path.join('../../src/'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import ResNet as net

# adjustables

k = 3#np.arange(11)                        # model index: should be in {0, 2, ..., 10}
dt = 0.01                     # time unit: 0.0005 for Lorenz and 0.01 for others
system = 'Cubic'         # system name: 'Hyperbolic', 'Cubic', 'VanDerPol', 'Hopf' or 'Lorenz'
noise = 0.0                   # noise percentage: 0.00, 0.01 or 0.02

lr = 1e-3                     # learning rate
max_epoch = 30000            # the maximum training epoch 
batch_size = 320              # training batch size
arch = [2, 128, 128, 128, 2]  # architecture of the neural network

# paths
data_dir = os.path.join('../../data/', system)
model_dir = os.path.join('../../models/', system)

# global const
n_forward_list = [1, 2, 3, 4, 5, 10, 15, 20]
step_size = 2**k

# load data
train_data = np.load(os.path.join(data_dir, 'train_noise{}.npy'.format(noise)))
val_data = np.load(os.path.join(data_dir, 'val_noise{}.npy'.format(noise)))
test_data = np.load(os.path.join(data_dir, 'test_noise{}.npy'.format(noise)))
n_train = train_data.shape[0]
n_val = val_data.shape[0]
n_test = test_data.shape[0]

for n_forward in n_forward_list:
    # create dataset object
    dataset = net.DataSet(train_data, val_data, test_data, dt, step_size, n_forward)

    model_name = 'model_D{}_noise{}_n_forward{}.pt'.format(step_size, noise, n_forward)

    # create/load model object
    print('create model {} ...'.format(model_name))
    model = net.ResNet(arch=arch, dt=dt, step_size=step_size)

    # training
    model.train_net(dataset, max_epoch=max_epoch, batch_size=batch_size, lr=lr,
                    model_path=os.path.join(model_dir, model_name))
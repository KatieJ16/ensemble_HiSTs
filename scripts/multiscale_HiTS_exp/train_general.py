"""
Trains the depend method with 3 timescales
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

module_path = os.path.abspath(os.path.join('../../src/'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import Resnet_multiscale_general as net

#===========================================================================================================

# adjustables

parser = argparse.ArgumentParser(description="""Add description.""")
parser.add_argument("-n", "--noise", help="level of noise", type=float)
parser.add_argument("-s", "--system", help="system type: VanDerPol, Lorenz, KS")
parser.add_argument("-l", "--letter", default='a')
parser.add_argument("-ss", "--small", default=0, type=int)
args = parser.parse_args()
print("noise = ", args.noise)
print("system = ", args.system)
print("letter = ", args.letter)
print("small = ", args.small)

system = args.system
letter = args.letter
noise = args.noise
print("system = ", system)

lr = 1e-3                     # learning rate
max_epoch = 500000            # the maximum training epoch 
batch_size = 320              # training batch size

combos_file = None


# paths
data_dir = os.path.join('../../data/', system,)
model_dir = os.path.join('../../models/', system)


# load data
train_data = np.load(os.path.join(data_dir, 'train_noise{}.npy'.format(noise)))
try:
    val_data = np.load(os.path.join(data_dir, 'val_noise{}.npy'.format(noise)))
    test_data = np.load(os.path.join(data_dir, 'test_noise{}.npy'.format(noise)))
except:
    #just use training data for all if validation and testing not there
    val_data= train_data
    test_data=train_data
    
n_train, _, ndim = train_data.shape
n_val = val_data.shape[0]
n_test = test_data.shape[0]

print("train_data shape = ", train_data.shape)
print("val_data shape = ", val_data.shape)
print("test_data.shape = ", test_data.shape)

n_forward = 5
override_step_list = True
if 'KS' in system:
    n_forward = 3
    override_step_list = False
    dt = 0.025
    arch = [ndim, 2048, ndim]
    
    if args.small == 1:
        step_sizes  = [1, 6, 36]
        combos_file = "all_combos_KS6.npy"
    else:
        step_sizes  = [6, 36, 216]
        combos_file = "all_combos_KS36.npy"
        
elif 'fluid' in system:
    n_forward = 3
    override_step_list = False
    dt = 0.025
    arch = [ndim, 256, ndim]
    step_sizes  = [1, 4, 16]
    combos_file = "all_combos_fluid4.npy"
    
elif 'flower' in system:
    n_forward = 3
    override_step_list = False
    dt = 0.025
    arch = [ndim, 128, ndim]
    step_sizes  = [1, 4, 16]
    combos_file = "all_combos_fluid4.npy"
    
elif 'Bach' in system:
    n_forward = 3
    override_step_list = False
    dt = 0.025
    arch = [ndim, 2048, ndim]
    step_sizes  = [1, 5, 25]
    combos_file = "all_combos_5.npy"
    print('combos_file = ', combos_file)
    
elif "VanDerPol" in system:
    smallest_step = 4
    dt = 0.01
    arch = [2, 512, 512, 512, 2]
    
elif "Lorenz" in system:
    lr = 1e-4  
    smallest_step = 16
    dt = 0.0005
    arch = [3, 1024, 1024, 1024, 3]
    
elif "hyperbolic" in system:
    lr = 1e-4  
    smallest_step = 8
    dt = 0.01
    arch = [2, 128, 128, 128, 2] 
    
elif "cubic" in system:
    lr = 1e-4  
    smallest_step = 2
    dt = 0.01
    arch = [2, 256, 256, 256, 2] 
    
elif "hopf" in system:
    lr = 1e-4  
    smallest_step = 4
    dt = 0.01
    arch = [3, 128, 128, 128, 3]
    
else:
    print("system not available")
    raise SystemExit()
    
if args.small > 0:
    smallest_step = args.small
    if override_step_list:
        step_sizes = [smallest_step, smallest_step*2, smallest_step*4]
    print("step_sizes= ", step_sizes)
    
    
print("step_sizes = ", step_sizes)
print("smallest_step = ", smallest_step)

n_poss = 1773


#make and train
model_name = 'model_{}_depends_stepsize{}_noise{}_{}.pt'.format(system, smallest_step, noise, letter)



# create/load model object
try:
    #load model if it exists
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(os.path.join(model_dir, model_name), map_location=device)
    model.device = device
    print("model loaded ", model_name)
except:
    # create model object if it doesnt wokr
    print('create model {} ...'.format(model_name))
    model = net.ResNet(arch=arch, dt=dt, step_sizes=step_sizes, n_poss=n_poss, combos_file = combos_file)


    # train large first, then all together (just training a little so it doesn't take ages
    #only train indiviual when we are making new object
    for i in  step_sizes:     
        print("step_size = ", i)
        dataset = net.DataSet(train_data, val_data, test_data, dt, i, n_forward)
        model.train_net_single(dataset, max_epoch=1000, batch_size=batch_size, lr=lr,
                        model_path=os.path.join(model_dir, model_name), print_every=100, type=str(i))
# training
n_forward = min(int(np.max(step_sizes)*4/np.min(step_sizes)) + 1, int(train_data.shape[1] / np.min(step_sizes))-1)
print("n_forward = ", n_forward)
dataset = net.DataSet(train_data, val_data, test_data, dt, smallest_step, n_forward)
model.train_net(dataset, max_epoch=max_epoch, batch_size=batch_size, lr=lr,
                model_path=os.path.join(model_dir, model_name), print_every=1)
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
# from tqdm.notebook import tqdm

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
# parser.add_argument("password", help="Password")
# parser.add_argument("email", help="Email")
# parser.add_argument("--gender", help="Gender")
args = parser.parse_args()
print("noise = ", args.noise)
print("system = ", args.system)
print("letter = ", args.letter)

system = args.system
letter = args.letter
print("system = ", system)
if system == 'KS' or system == "KS_new":
    smallest_step = 4#1
    step_sizes  = [4, 8, 216]
    dt = 0.025
    arch = [512, 2048, 512]
    combos_file = "all_combos_KS.npy"
elif system == "VanDerPol":
    smallest_step = 4
    dt = 0.01
    arch = [2, 512, 512, 512, 2]
elif system == "Lorenz":
    smallest_step = 16
    dt = 0.0005
    arch = [3, 1024, 1024, 1024, 3]
else:
    print("system not available")
    raise SystemExit()
    
n_poss = 1773
    
# smallest_step = 1#64
# dt = 0.0250

noise = args.noise
# for noise in [0.0]:#, 0.01, 0.02, 0.05, 0.1, 0.2]:

lr = 1e-3                     # learning rate
max_epoch = 500000            # the maximum training epoch 
batch_size = 320              # training batch size
# arch = [512, 2048, 512]  # architecture of the neural network, KS


# paths
data_dir = os.path.join('../../data/', system,)
model_dir = os.path.join('../../models/', system)#, '{}_{}'.format(system, str(n_poss),)) #'../../models/'  VanDerPol'
#     model_dir = '../../models/VanDerPol_multiscale'
# model_dir = '../../models/KS_'+str(n_poss)


# load data
# train_data = np.load(os.path.join(data_dir, 'train_long_noise{}.npy'.format(noise)))
train_data = np.load(os.path.join(data_dir, 'train_noise{}.npy'.format(noise)))
try:
    val_data = np.load(os.path.join(data_dir, 'val_noise{}.npy'.format(noise)))
    test_data = np.load(os.path.join(data_dir, 'test_noise{}.npy'.format(noise)))
except:
    #just use training data for all if validation and testing not there
    val_data= train_data
    test_data=train_data
    
n_train = train_data.shape[0]
n_val = val_data.shape[0]
n_test = test_data.shape[0]

print("train_data shape = ", train_data.shape)
print("val_data shape = ", val_data.shape)
print("test_data.shape = ", test_data.shape)


# for step_size in [smallest_step]:#,8,16,32]:

#     for letter in ['a']:
    #make and train
model_name = 'model_{}_depends{}_stepsize{}_noise{}_{}.pt'.format(system, n_poss, smallest_step, noise, letter)

#define step_sizes if not already done
try:
    print("step_sizes= ", step_sizes)
except:
    step_sizes = [smallest_step, smallest_step*2, smallest_step*4]
    print("step_sizes= ", step_sizes)

# # create/load model object
try:
#     ghj
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(os.path.join(model_dir, model_name), map_location=device)
    model.device = device
    print("model loaded ", model_name)
except:
    # create/load model object
    print('create model {} ...'.format(model_name))
    model = net.ResNet(arch=arch, dt=dt, step_sizes=step_sizes, n_poss=n_poss, combos_file = combos_file)#, prev_models=prev_models)


#             train large first, then all together (just training a little so it doesn't take ages
    #only train indiviual when we are making new object
    for i in  step_sizes:     
        print("step_size = ", i)
        n_forward = 5
        dataset = net.DataSet(train_data, val_data, test_data, dt, i, n_forward)
        model.train_net_single(dataset, max_epoch=100, batch_size=batch_size, lr=lr,
                        model_path=os.path.join(model_dir, model_name), print_every=100, type=str(i))
    # training
n_forward = int(np.max(step_sizes)*4/np.min(step_sizes)) + 1
print("n_forawrd = ", n_forward)
dataset = net.DataSet(train_data, val_data, test_data, dt, smallest_step, n_forward)
model.train_net(dataset, max_epoch=max_epoch, batch_size=batch_size, lr=lr,
                model_path=os.path.join(model_dir, model_name), print_every=10)
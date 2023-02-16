import os
import sys
import torch
import numpy as np
 
module_path = os.path.abspath(os.path.join('../../src/'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import ResNet as net
import argparse

parser = argparse.ArgumentParser(description="""Add description.""")
parser.add_argument("-n", "--noise", help="level of noise", type=float)
parser.add_argument("-ss", "--step_size", help="step size", type=int)
parser.add_argument("-s", "--system", help="system, KS (KS_new), Lorenz, VanDerPol")
parser.add_argument("-l", "--letter", default='a')
# parser.add_argument("password", help="Password")
# parser.add_argument("email", help="Email")
# parser.add_argument("--gender", help="Gender")
args = parser.parse_args()
noise = args.noise
step_size = args.step_size
letter = args.letter
system = args.system
print("step_size = ", step_size)
print("system = ", system)
print("system = ", args.system)
print("letter = ", args.letter)

# adjustables

# k = 1                        # model index: should be in {0, 2, ..., 10}
# dt = 0.01                     # time unit: 0.0005 for Lorenz and 0.01 for others
# system = 'KS_new'         # system name: 'Hyperbolic', 'Cubic', 'VanDerPol', 'Hopf' or 'Lorenz'
# noise = 0.0                   # noise percentage: 0.00, 0.01 or 0.02

lr = 1e-3                     # learning rate
max_epoch = 100000            # the maximum training epoch 
batch_size = 320              # training batch size
# arch = [2, 128, 128, 128, 2]  # architecture of the neural network

backward = False
if system == 'KS' or system == "KS_new":
#     smallest_step = 1
    dt = 0.025
    if step_size > 36:
        arch = [512, 512, 512]
    else:
        arch = [512, 2048, 512]
        
    train_data_file = 'train_long_noise{}.npy'.format(noise)
        
elif system == "VanDerPol":
#     smallest_step = 4
    dt = 0.01
    arch = [2, 512, 512, 512, 2]
    
elif system == "VanDerPol_backward":
    data_dir = '../../data/VanDerPol'
#     smallest_step = 4
    dt = 0.01
    arch = [2, 512, 512, 512, 2]
    backward = True
elif system == "Lorenz":
#     smallest_step = 16
    dt = 0.0005
    arch = [3, 1024, 1024, 1024, 3]
else:
    print("system not available")
    raise SystemExit()

    
# paths
try:
    print(data_dir)
except:
    data_dir = os.path.join('../../data/', system)
    
model_dir = os.path.join('../../models/', system)

# global const
n_forward = 5
# step_size = 2**k

# load data
try:
    train_data = np.load(os.path.join(data_dir, train_data_file))
except:
    train_data = np.load(os.path.join(data_dir, 'train_noise{}.npy'.format(noise)))
    
try:
    val_data = np.load(os.path.join(data_dir, 'val_noise{}.npy'.format(noise)))
except:
    print("no validation found, using training.")
    val_data = train_data
    
try:
    test_data = np.load(os.path.join(data_dir, 'test_noise{}.npy'.format(noise)))
except:
    print("no testing found, using training.")
    test_data = train_data
n_train = train_data.shape[0]
n_val = val_data.shape[0]
n_test = test_data.shape[0]
    
    

# create dataset object
dataset = net.DataSet(train_data, val_data, test_data, dt, step_size, n_forward, backward=backward)

model_name = 'original_model_D{}_noise{}_{}.pt'.format(step_size, noise, letter)

# create/load model object
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(os.path.join(model_dir, model_name), map_location=device)
    model.device = device
except:
    print('create model {} ...'.format(model_name))
    model = net.ResNet(arch=arch, dt=dt, step_size=step_size)

# training
model.train_net(dataset, max_epoch=max_epoch, batch_size=batch_size, lr=lr,
                model_path=os.path.join(model_dir, model_name))

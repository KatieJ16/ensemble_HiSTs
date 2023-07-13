#does postprocessing for both original and new method on same graphs


import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time
import scipy
import random
import argparse

module_path = os.path.abspath(os.path.join('../../src/'))
if module_path not in sys.path:
    sys.path.append(module_path)

import Resnet_multiscale_general as net


parser = argparse.ArgumentParser(description="""Add description.""")
parser.add_argument("-n", "--noise", help="level of noise", type=float)
parser.add_argument("-s", "--system", help="system, KS (KS_new), Lorenz, VanDerPol")
parser.add_argument("-l", "--letter", default='a')
parser.add_argument("-ss", "--small", default=0, type=int)
args = parser.parse_args()
noise = args.noise
letter = args.letter
system = args.system
print("system = ", system)
print("letter = ", letter)
print("noise = ", noise)
print("small = ", args.small)


#===========================================================================================================
def find_ave_random_paths(t_list_list, mse_list):
    to_ave = np.zeros((num_lines, 10000, test_data.shape[-1]))
    i = 0 
    j = 4
    print(t_list_list[i][j])
    for i in range(num_lines):
        for j in range(len(t_list_list[i])):
            to_ave[i, int(t_list_list[i][j])] = mse_list[i][j]

    averages = np.zeros(10000)
    to_ave[to_ave == 0] = np.nan
    means = np.nanmean(to_ave[:, 1:], axis=0)
    stds = np.nanstd(to_ave[:, 1:], axis=0)

    mask = np.isfinite(means[:,0])
    ts = np.arange(1, 10000)

    return ts[mask], means[mask], stds[mask]
   
#===========================================================================================================

def predict_random_combo(model_depends, models_original, test_data, timesteps = 5000, to_plot=True):
    t = 0
    n_test_points, _, ndim = test_data.shape


    t_list = list()
    y_pred_list_depends = list()
    y_pred_list_original = list()

    indices = np.random.randint(0,len(step_sizes), int(timesteps/min(step_sizes)))
    steps = list()
    for i in range(len(indices)):
        steps.append(step_sizes[indices[i]])
        if sum(steps)>timesteps:
            break
    
    y_pred_list_depends = np.zeros((n_test_points, len(steps)-1, ndim))
    y_pred_list_original = np.zeros((n_test_points, len(steps)-1, ndim))
    y_preds_depends = torch.tensor(test_data[:, 0]).float()
    y_preds_original = torch.tensor(test_data[:, 0]).float()

    for i in range(len(steps)-1):
        this_pick = indices[i]
        this_step_size = steps[i]
        t+= this_step_size
        y_preds_depends = model_depends.forward(y_preds_depends, str(this_step_size))
        y_preds_original = models_original[this_pick].forward(y_preds_original)

        y_pred_list_depends[:,i] = y_preds_depends.detach().numpy()
        y_pred_list_original[:,i] = y_preds_original.detach().numpy()
        t_list.append(t)
        
    t_list = np.array(t_list)
    y_pred_list_depends = np.array(y_pred_list_depends)
    y_pred_list_original = np.array(y_pred_list_original)
    
    
    mse_depends = np.mean((y_pred_list_depends - test_data[:,t_list.astype(int)])**2, axis = (0,2))
    mse_original = np.mean((y_pred_list_original - test_data[:,t_list.astype(int)])**2, axis = (0,2))
    
    return y_pred_list_depends, y_pred_list_original, mse_depends, mse_original, t_list
#===========================================================================================================

spacial = False

override_step_list = True
step_sizes = [4, 8, 216]#[4, 8, 216]
# timesteps = 5000
if 'KS' in system:
    override_step_list = False
    dt = 0.025
    arch = [512, 2048, 512]
    step_sizes = [1, 6, 36]#[4, 8, 216]
    spacial = True
    
elif 'fluid' in system:
    n_forward = 3
    override_step_list = False
    dt = 0.01
    arch = [22, 256, 22]
    step_sizes  = [1, 4, 16]
    combos_file = "all_combos_fluid4.npy"
    
elif  "VanDerPol" in system:
    smallest_step = 4
    dt = 0.01
    arch = [2, 512, 512, 512, 2]
    step_sizes = [4, 8, 16]
elif "Lorenz" in system:
    smallest_step = 16
    timesteps = 1000
    dt = 0.0005
    arch = [3, 1024, 1024, 1024, 3]
    step_sizes = [16, 32, 64]
    
elif "hyperbolic" in system:
    lr = 1e-4  
    smallest_step = 8
    step_sizes = [8, 16, 32]
    dt = 0.01
    arch = [2, 128, 128, 128, 2] 
elif  "cubic" in system:
    lr = 1e-4  
    smallest_step = 2
    dt = 0.01
    arch = [2, 256, 256, 256, 2] 
    step_sizes = [2, 4, 8]
elif "hopf" in system:
    lr = 1e-4  
    smallest_step = 4
    dt = 0.01
    arch = [3, 128, 128, 128, 3]
    step_sizes = [4, 8, 16]
else:
    print("system not available")
    raise SystemExit()

if args.small > 0:
    smallest_step = args.small
    if override_step_list:
        step_sizes = [smallest_step, smallest_step*2, smallest_step*4]
    
n_poss = 1773

# paths
data_dir = os.path.join('../../data/', system,)
model_dir = os.path.join('../../models/', system)


# load data
train_data = np.load(os.path.join(data_dir, 'train_noise{}.npy'.format(noise)))
try:
    val_data = np.load(os.path.join(data_dir, 'val_noise{}.npy'.format(noise)))
except:
    #just use training data for all if validation and testing not there
    print("validation not found, using trainin")
    val_data= train_data
try:
    test_data = np.load(os.path.join(data_dir, 'test_noise{}.npy'.format(noise)))
except:
    #just use training data for all if validation and testing not there
    print("testing not found, using training ")
    test_data=train_data


n_test_points, _, ndim = test_data.shape

n_train = train_data.shape[0]
n_val = val_data.shape[0]
n_test = test_data.shape[0]

print("train_data shape = ", train_data.shape)
print("val_data shape = ", val_data.shape)
print("test_data.shape = ", test_data.shape)




#load depend model
model_name = 'model_{}_depends_stepsize{}_noise{}_{}.pt'.format(system, smallest_step, noise, letter)
model_depends = torch.load(os.path.join(model_dir, model_name), map_location='cpu')
model_depends.device = 'cpu'

print("model_depends.step_sizes= ", model_depends.step_sizes)
for step_size in model_depends.step_sizes:
    model_depends._modules[str(step_size)]._modules['activation'] = torch.nn.ReLU()

    
#load original models
# load models
models_original = list()
for step_size in model_depends.step_sizes:
    model_name = 'original_model_D{}_noise{}.pt'.format(step_size, noise, letter)
    print("load ", model_name)
    try:
        try:
            models_original.append(torch.load(os.path.join(model_dir, model_name), map_location='cpu'))
        except:
            try:
                model_name = 'original_model_D{}_noise{}_{}.pt'.format(step_size, noise, letter)
                models_original.append(torch.load(os.path.join(model_dir, model_name), map_location='cpu'))
            except:
                model_name = 'original_model_D{}_noise{}_{}.pt'.format(step_size, noise, 'b')
                models_original.append(torch.load(os.path.join(model_dir, model_name), map_location='cpu'))
    except:
        print("done loading at {}".format(step_size))
        break
        

to_plot = False
timesteps = test_data.shape[1] - 1
#less for Lorenz
if "Lorenz" in system:
    timesteps = 1000
    
elif "KS" in system:
    timesteps = 1000
    
print("timesteps = ", timesteps)

        
num_lines = 100
mse_list_depends = list()
mse_list_original = list()
t_list_list = list()
path_list = list()
y_pred_list_depends = list()
y_pred_list_original = list()


#train all the random lines of both models
for i in range(num_lines):
    print("i = ", i)

    y_preds_random_depends, y_preds_random_original, mse_random_depends, mse_random_original, t_list_random = predict_random_combo(model_depends, models_original, test_data, timesteps = timesteps, to_plot=False)
    y_pred_list_depends.append(y_preds_random_depends[0,:])
    y_pred_list_original.append(y_preds_random_original[0,:])
    mse_list_depends.append(mse_random_depends)
    mse_list_original.append(mse_random_original)
    t_list_list.append(t_list_random)

#find the average paths for both methods
ts_depends, means_depends, stds_depends = find_ave_random_paths(t_list_list, mse_list_depends)
ts_original, means_original, stds_original = find_ave_random_paths(t_list_list, mse_list_original)


#####################################################################################################
#start of the plotting code

# graph the mse both themods with a random line for each trial and thicker line for average of mse
plt.figure()
for i in range(len(mse_list_depends)):
    plt.semilogy(t_list_list[i], mse_list_depends[i], 'r', linewidth = 0.25, alpha = 0.5)
    plt.semilogy(t_list_list[i], mse_list_original[i], 'b', linewidth = 0.25, alpha=0.5)

plt.semilogy(ts_depends, means_depends[:,0], color='darkred', label="depends")
plt.semilogy(ts_original, means_original[:,0], color='navy', label="original")
plt.legend()
plt.title(system + ": MSE : noise = " + str(noise) + ": " + str(num_lines) + " random paths :step_sizes = "+str(step_sizes))
plt.savefig("{}_{}_min{}_MSE_both_n{}_all.jpg".format(system, letter, min(step_sizes), noise))


#make plot of just the first predicted line
#when inputs is 3 or more, plot each dim on a new plot
if ndim < 3:
    plt.figure()
    plt.plot(test_data[0,:timesteps], 'g', linewidth = 0.5)
    for i in range(len(t_list_list)):
        plt.plot(t_list_list[i], y_pred_list_depends[i], 'r', linewidth = 0.25, alpha = 0.05)
        plt.plot(t_list_list[i], y_pred_list_original[i], 'b', linewidth = 0.25, alpha = 0.05)

    plt.title(system + ": Predicted 1 path : noise = " + str(noise) +" :step_sizes = "+str(step_sizes))
    plt.savefig("{}_{}_min{}_both_predict_first_n{}.jpg".format(system, letter, min(step_sizes), noise))
else:
    idx = 1
    print("y_pred_list_depends[i] shape = ", y_pred_list_depends[0].shape)
    plt.figure()
    plt.plot(test_data[0,:timesteps, idx], 'g', linewidth = 0.5)
    for i in range(len(t_list_list)):
        plt.plot(t_list_list[i], y_pred_list_depends[i][:,idx], 'r', linewidth = 0.5, alpha = 0.5)
        plt.plot(t_list_list[i], y_pred_list_original[i][:,idx], 'b', linewidth = 0.5, alpha = 0.5)
        
    plt.ylim([np.min(test_data[0,:,idx]), np.max(test_data[0,:,idx])])
    plt.title(system + ": Predicted 1 path, idx = "+str(idx)+": noise = " + str(noise) +" :step_sizes = "+str(step_sizes))
    plt.savefig("{}_{}_min{}_both_predict_first_n{}.jpg".format(system, letter, min(step_sizes), noise))


# want a plot with of the first test plot. Where the predicted has average of all paths in thick line and shaded for 1 and 2 stds. 

n_timesteps = 1e10
for i in range(len(t_list_list)):
    if n_timesteps > max(t_list_list[i]):
        n_timesteps = max(t_list_list[i])
        
y_preds_all_depends = np.zeros((len(y_pred_list_depends), n_timesteps, train_data.shape[2]))
y_preds_all_original = np.zeros((len(y_pred_list_original), n_timesteps, train_data.shape[2]))
#we need to interpolate y_preds
for i in range(len(t_list_list)):
    t = np.insert(t_list_list[i], 0, 0)
    y_depends = np.insert(y_pred_list_depends[i], 0, test_data[0,0], axis = 0)
    y_original = np.insert(y_pred_list_original[i], 0, test_data[0,0], axis = 0)
    
    sample_steps = range(0, n_timesteps)
    cs_depends = scipy.interpolate.interp1d(t, y_depends.T, kind='linear')
    y_preds_depends = cs_depends(sample_steps).T
    y_preds_all_depends[i] = y_preds_depends
    
    cs_original = scipy.interpolate.interp1d(t, y_original.T, kind='linear')
    y_preds_original = cs_original(sample_steps).T
    y_preds_all_original[i] = y_preds_original


means_depends = np.mean(y_preds_all_depends, axis = 0)
stds_depends = np.std(y_preds_all_depends, axis = 0)

means_original = np.mean(y_preds_all_original, axis = 0)
stds_original = np.std(y_preds_all_original, axis = 0)

ts = sample_steps


#plot the means and stds
plt.figure()
plt.plot(test_data[0,:timesteps], 'g', label = 'test_data', linewidth = 0.5)
plt.plot(ts, means_depends, 'r', label = "depends")
plt.plot(ts, means_original, 'blue', label = "original")
for i in range(len(means_depends[0])):
    for n_stds in range(1, 4):
        plt.fill_between(ts, means_depends[:,i]+n_stds*stds_depends[:,i], means_depends[:,i]-n_stds*stds_depends[:,i], facecolor='red', alpha=0.25)
        plt.fill_between(ts, means_original[:,i]+n_stds*stds_original[:,i], means_original[:,i]-n_stds*stds_original[:,i], facecolor='blue', alpha=0.25)
        
plt.legend()
plt.title(system + ": Predicted 1 path: noise = " + str(noise) +" :step_sizes = "+str(step_sizes))
plt.savefig("{}_{}_min{}_both_predict_first_shaded_n{}.jpg".format(system, letter, min(step_sizes), noise))

#make shaded plots on 3
if ndim == 3:
    fig, axs = plt.subplots(3, 1)
    for i in range(3):
        #plot the means and stds
        axs[i].plot(test_data[0,:timesteps, i], 'g', label = 'test_data', linewidth = 0.5)
        axs[i].plot(ts, means_depends[:,i], 'r', label = "depends")
        axs[i].plot(ts, means_original[:,i], 'blue', label = "original")
        for n_stds in range(1, 4):
            axs[i].fill_between(ts, means_depends[:,i]+n_stds*stds_depends[:,i], 
                             means_depends[:,i]-n_stds*stds_depends[:,i], facecolor='red', alpha=0.25)
            axs[i].fill_between(ts, means_original[:,i]+n_stds*stds_original[:,i], 
                             means_original[:,i]-n_stds*stds_original[:,i], facecolor='blue', alpha=0.25)
        
    plt.suptitle(system + ": Predicted 1 path: noise = " + str(noise) +" :step_sizes = "+str(step_sizes))
    fig.tight_layout()
    plt.savefig("{}_{}_min{}_both_predict_first_shaded_3plots_n{}.jpg".format(system, letter, min(step_sizes), noise))
    
    
    #plot the on with each line too
    fig, axs = plt.subplots(3, 1)
    for i_ndim in range(3):
        axs[i_ndim].plot(test_data[0,:timesteps, i_ndim], 'g', linewidth = 0.5)
        for i in range(len(t_list_list)):
            axs[i_ndim].plot(t_list_list[i], y_pred_list_depends[i][:,i_ndim], 'r', linewidth = 0.25, alpha = 0.05)
            axs[i_ndim].plot(t_list_list[i], y_pred_list_original[i][:, i_ndim], 'b', linewidth = 0.25, alpha = 0.05)

    plt.suptitle(system + ": Predicted 1 path : noise = " + str(noise) +" :step_sizes = "+str(step_sizes))
    fig.tight_layout()
    plt.savefig("{}_{}_min{}_both_predict_first_3plots_n{}.jpg".format(system, letter, min(step_sizes), noise))



#plot just 1
if ndim < 3:
    plt.figure()
    plt.plot(test_data[0,:timesteps], 'g', label = 'test_data')
    # plt.plot(ts, means_depends, 'r-', label = "depends means")
    # plt.plot(ts, means_original, 'b-', label = "original means")
    plt.plot(t_list_list[0], y_pred_list_depends[0], 'r', label = "depends 1 path")
    plt.plot(t_list_list[0], y_pred_list_original[0], 'b', label = "original 1 path")
else:
    plt.figure()
    plt.plot(test_data[0,:timesteps, idx], 'g', label = 'test_data')
    # plt.plot(ts, means_depends, 'r-', label = "depends means")
    # plt.plot(ts, means_original, 'b-', label = "original means")
    plt.plot(t_list_list[0], y_pred_list_depends[0][ :,idx], 'r', label = "depends 1 path")
    plt.plot(t_list_list[0], y_pred_list_original[0][ :, idx], 'b', label = "original 1 path")
    

plt.ylim([np.min(test_data[0,:,idx]), np.max(test_data[0,:,idx])])
# plt.legend()
plt.title(system + ": Predicted 1 path: noise = " + str(noise) +" :step_sizes = "+str(step_sizes))
plt.savefig("{}_{}_min{}_both_1_path_n{}.jpg".format(system, letter, min(step_sizes), noise))


#make spacial figures
if spacial == True:
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    vmin = np.min(test_data)
    vmax = np.max(test_data)
    axs[0].imshow(test_data[0,:timesteps], vmin=vmin, vmax=vmax)
    axs[1].imshow(y_preds_all_original[0,:timesteps], vmin=vmin, vmax=vmax)
    axs[2].imshow(y_preds_all_depends[0,:timesteps], vmin=vmin, vmax=vmax)
    
    
    fig.suptitle(system + ": noise = " + str(noise) +" :step_sizes = "+str(step_sizes))
    plt.savefig("{}_{}_min{}_spacial_plots_n{}.jpg".format(system, letter, min(step_sizes), noise))


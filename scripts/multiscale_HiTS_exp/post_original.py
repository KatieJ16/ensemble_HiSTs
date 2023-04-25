import os
import sys
import time
import torch
import numpy as np
import scipy.interpolate
# from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse


module_path = os.path.abspath(os.path.join('../../src/'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import ResNet as net

#===========================================================================================================
parser = argparse.ArgumentParser(description="""Add description.""")
parser.add_argument("-n", "--noise", help="level of noise", type=float)
parser.add_argument("-s", "--system", help="system, KS (KS_new), Lorenz, VanDerPol")
parser.add_argument("-l", "--letter", default='a')
args = parser.parse_args()
noise = args.noise
letter = args.letter
system = args.system
print("system = ", system)
print("letter = ", letter)
print("noise = ", noise)

#===========================================================================================================

# adjustables
dt = 0.01                     # time unit: 0.0005 for Lorenz and 0.01 for others
# system = 'VanDerPol'         # system name: 'Hyperbolic', 'Cubic', 'VanDerPol', 'Hopf' or 'Lorenz'

# path
data_dir = os.path.join('../../data/', system)
model_dir = os.path.join('../../models/', system)

# global const
# ks = [1]#list(range(11))
# step_sizes = [2**k for k in ks]
#step_sizes = [4, 8, 16]#[4,  8, 216]

if system == 'KS' or system == "KS_new":
    smallest_step = 4
    dt = 0.025
    arch = [512, 2048, 512]
    step_sizes = [4, 8, 216]
elif system == "VanDerPol":
    smallest_step = 4
    dt = 0.01
    arch = [2, 512, 512, 512, 2]
    step_sizes = [4, 8, 16]
elif system == "Lorenz":
    smallest_step = 16
    dt = 0.0005
    arch = [3, 1024, 1024, 1024, 3]
    step_sizes = [1, 2, 4, 8, 16, 32, 64]
    
elif system == "hyperbolic":
    dt = 0.01
    arch = [2, 128, 128, 128, 2] 
    step_sizes = [8, 16, 32]#[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
elif "cubic" in system:
    dt = 0.01
    arch = [2, 256, 256, 256, 2] 
    step_sizes = [8, 16, 32]#[2, 4, 8]# [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
elif "hopf" in system:
    dt = 0.01
    arch = [3, 128, 128, 128, 3]
    step_sizes = [4, 8, 16]#[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
else:
    print("system not available")
    raise SystemExit()
    
# letter = 'a'
# noise = 0.1

# path
data_dir = os.path.join('../../data/', system)
model_dir = os.path.join('../../models/', system)


# load validation set and test set
val_data = np.load(os.path.join(data_dir, 'val_noise{}.npy'.format(noise)))
test_data = np.load(os.path.join(data_dir, 'test_noise{}.npy'.format(noise)))
perfect_data  = np.load(os.path.join(data_dir, 'test_noise{}.npy'.format(0.0)))

# load models
models = list()
for step_size in step_sizes:
    model_name = 'original_model_D{}_noise{}.pt'.format(step_size, noise, letter)
    print("load ", model_name)
    try:
        try:
            models.append(torch.load(os.path.join(model_dir, model_name), map_location='cpu'))
        except:
            try:
                model_name = 'original_model_D{}_noise{}_{}.pt'.format(step_size, noise, letter)
                models.append(torch.load(os.path.join(model_dir, model_name), map_location='cpu'))
            except:
                model_name = 'original_model_D{}_noise{}_{}.pt'.format(step_size, noise, 'b')
                models.append(torch.load(os.path.join(model_dir, model_name), map_location='cpu'))
    except:
        print("done loading at {}".format(step_size))
        break
        


# fix model consistencies trained on gpus (optional)
for model in models:
    model.device = 'cpu'
    model._modules['increment']._modules['activation'] = torch.nn.ReLU()


# shared info
n_steps = test_data.shape[1] - 1
t = [dt*(step+1) for step in range(n_steps)]
criterion = torch.nn.MSELoss(reduction='none')


# uniscale time-stepping with NN
preds_mse = list()
times = list()

for model in models:
    start = time.time()
    y_preds = model.uni_scale_forecast(torch.tensor(test_data[:, 0, :]).float(), n_steps=n_steps)
    
    plt.figure()
    plt.plot(y_preds[0, :500], label = "y_preds")
    plt.plot(test_data[0, 1:501, ], label = "truth")
    plt.legend()
    plt.title(str(model.step_size))
    plt.savefig("{}_single_{}.pdf".format(system, model.step_size))
    
    end = time.time()
    times.append(end - start)
    preds_mse.append(criterion(torch.tensor(test_data[:, 1:, :]).float(), y_preds).mean(-1))
    

# visualize forecasting error at each time step    
fig = plt.figure(figsize=(30, 10))
colors=iter(plt.cm.rainbow(np.linspace(0, 1, len(preds_mse))))
# multiscale_err = multiscale_preds_mse.mean(0).detach().numpy()
for k in range(len(preds_mse)):
    print(k)
    err = preds_mse[k]
    mean = err.mean(0).detach().numpy()
    rgb = next(colors)
    plt.semilogy(t, mean, linestyle='-', color=rgb, linewidth=5, label='$\Delta\ t$={}dt'.format(step_sizes[k]))
# plt.semilogy(t, multiscale_err, linestyle='-', color='k', linewidth=6, label='multiscale')
plt.legend(fontsize=30, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.2))
plt.xticks(fontsize=60)
plt.yticks(fontsize=60)


# plt.ylim([0.001, 10])

plt.savefig("{}_original_singles_mse_n{}.pdf".format(system, noise))

# multiscale time-stepping with NN
start = time.time()
y_preds = net.vectorized_multi_scale_forecast(torch.tensor(test_data[:, 0, :]).float(), n_steps=n_steps, models=models)
end = time.time()
multiscale_time = end - start
multiscale_preds_mse = criterion(torch.tensor(test_data[:, 1:, :]).float(), y_preds).mean(-1)

# visualize forecasting error at each time step    
fig = plt.figure(figsize=(30, 10))
colors=iter(plt.cm.rainbow(np.linspace(0, 1, len(preds_mse))))
multiscale_err = multiscale_preds_mse.mean(0).detach().numpy()
for k in range(len(preds_mse)):
    print(k)
    err = preds_mse[k]
    mean = err.mean(0).detach().numpy()
    rgb = next(colors)
    plt.semilogy(t, mean, linestyle='-', color=rgb, linewidth=5, label='$\Delta\ t$={}dt'.format(step_sizes[k]))
plt.semilogy(t, multiscale_err, linestyle='-', color='k', linewidth=6, label='multiscale')
plt.legend(fontsize=30, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.2))
plt.xticks(fontsize=60)
plt.yticks(fontsize=60)

# plt.ylim([0.001, 10])

plt.savefig("{}_original_mse_n{}.pdf".format(system, noise))

import random
all_combos = np.load('all_combos_'+str(1)+'.npy', allow_pickle=True)
print(all_combos)
# all_combos = all_combos - 1

def predict_random_combo(models, test_data=test_data, timesteps = 5000, to_plot=True):


#     step_sizes = [8, 16]
    t = 0
    n_test_points, _, ndim = test_data.shape


    t_list = list()
    y_pred_list = list()

    indices = np.random.randint(0,len(step_sizes), int(timesteps/min(step_sizes)))
    steps = list()
    for i in range(len(indices)):
        steps.append(step_sizes[indices[i]])
        if sum(steps)>timesteps:
            break
    
    y_pred_list = np.zeros((n_test_points, len(steps)-1, ndim))
    y_preds = torch.tensor(test_data[:, 0]).float()

    for i in range(len(steps)-1):
        this_pick = indices[i]
        this_step_size = steps[i]
        t+= this_step_size
        y_preds = models[this_pick].forward(y_preds)

        y_pred_list[:,i-1] = y_preds.detach().numpy()
        t_list.append(t)
        
    t_list = np.array(t_list)
    y_pred_list = np.array(y_pred_list)
    
    if to_plot:
        plt.plot(t_list, y_pred_list[0,:,0])
        plt.plot(t_list, test_data[0,t_list.astype(int), 0])
        plt.title("step_size = " + str(step_size)+ ": noise = "+ str(noise))
        plt.show()
    
    mse = np.mean((y_pred_list - test_data[:,t_list.astype(int)])**2, axis = (0,2))
    if to_plot:
        plt.semilogy(t_list, mse)
        plt.title("step_size = " + str(step_size)+ ": noise = "+ str(noise))
        plt.show()
    
    return y_pred_list, mse, t_list#, path

num_lines = 500
mse_list = list()
t_list_list = list()
path_list = list()
y_pred_list = list()

for i in range(num_lines):
    print("i = ", i)
    y_preds_random, mse_random, t_list_random = predict_random_combo(models, to_plot = False,  timesteps = 5000)
    y_pred_list.append(y_preds_random[0,:])
    mse_list.append(mse_random)
    t_list_list.append(t_list_random)
    
def find_ave_random_paths(t_list_list, mse_list):
    to_ave = np.zeros((num_lines, 10000, test_data.shape[-1]))
    print(to_ave.shape)

    for i in range(num_lines):
        for j in range(len(t_list_list[i])):
            to_ave[i, t_list_list[i][j]] = mse_list[i][j]

    averages = np.zeros(10000)
    to_ave[to_ave == 0] = np.nan
    means = np.nanmean(to_ave[:, 1:], axis=0)

    mask = np.isfinite(means[:,0])
    ts = np.arange(1, 10000)
    
    return ts[mask], means[mask]

plt.figure()
for i in range(len(mse_list)):
    plt.semilogy(t_list_list[i], mse_list[i], 'r', linewidth = 0.5)#, label = path_list[i])

plt.semilogy(t_list_list[i], mse_list[i], 'r', linewidth = 0.5, label="random")


for k in range(len(preds_mse)):
    print(k)
    err = preds_mse[k]
    mean = err.mean(0).detach().numpy()
    plt.semilogy(mean, label='$\Delta\ t$={}dt'.format(step_sizes[k]))
    
plt.semilogy(multiscale_err, 'k', label = "multiscale")

ts, means = find_ave_random_paths(t_list_list, mse_list)
plt.plot(ts, means[:,0],  label = "average")


plt.legend()
plt.title(system + ": MSE : noise = " + str(noise) + ": " + str(num_lines) + " random paths, original: step_sizes = "+ str(step_sizes))
plt.savefig("{}_MSE_n{}_all_original_smallest{}.pdf".format(system, noise, min(step_sizes)))

plt.figure()
for i in range(len(y_pred_list)):
    plt.plot(t_list_list[i], y_pred_list[i], 'r')
plt.plot(test_data[0,1:], 'b')
plt.title(system + ": Predicted 1 path : noise = " + str(noise) +" original : step_sizes = "+ str(step_sizes))
plt.savefig("{}_predict_first_n{}_original_smallest{}.pdf".format(system, noise, min(step_sizes)))

# plt.figure()
# for i in range(len(y_pred_list)):
#     plt.plot(t_list_list[i], y_pred_list[i], 'r')
# plt.plot(test_data[0,1:], 'b')
# plt.xlim([3000, test_data.shape[1]])
# plt.title(system + ": Predicted 1 path zoomed: noise = " + str(noise)+" original")
# plt.savefig("{}_predict_first_zoomed_n{}_original.pdf".format(system, noise))

# #===========================================================================================================
# #repeat with perfect data


# # uniscale time-stepping with NN
# preds_mse = list()
# times = list()

# for model in models:
#     start = time.time()
#     y_preds = model.uni_scale_forecast(torch.tensor(test_data[:, 0, :]).float(), n_steps=n_steps)
    
#     end = time.time()
#     times.append(end - start)
#     preds_mse.append(criterion(torch.tensor(test_data[:, 1:, :]).float(), y_preds).mean(-1))
    

# # multiscale time-stepping with NN
# start = time.time()
# y_preds = net.vectorized_multi_scale_forecast(torch.tensor(test_data[:, 0, :]).float(), n_steps=n_steps, models=models)
# end = time.time()
# multiscale_time = end - start
# multiscale_preds_mse = criterion(torch.tensor(test_data[:, 1:, :]).float(), y_preds).mean(-1)

# # visualize forecasting error at each time step    
# fig = plt.figure(figsize=(30, 10))
# colors=iter(plt.cm.rainbow(np.linspace(0, 1, len(preds_mse))))
# multiscale_err = multiscale_preds_mse.mean(0).detach().numpy()
# for k in range(len(preds_mse)):
#     print(k)
#     err = preds_mse[k]
#     mean = err.mean(0).detach().numpy()
#     rgb = next(colors)
#     plt.semilogy(t, mean, linestyle='-', color=rgb, linewidth=5, label='$\Delta\ t$={}dt'.format(step_sizes[k]))
# plt.semilogy(t, multiscale_err, linestyle='-', color='k', linewidth=6, label='multiscale')
# plt.legend(fontsize=30, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.2))
# plt.xticks(fontsize=60)
# plt.yticks(fontsize=60)

# # plt.ylim([0.001, 10])

# plt.savefig("{}_original_mse_n{}_perfect.pdf".format(system, noise))


# num_lines = 500
# mse_list = list()
# t_list_list = list()
# path_list = list()
# y_pred_list = list()

# for i in range(num_lines):
#     y_preds_random, mse_random, t_list_random = predict_random_combo(models, to_plot = False,  timesteps = 5000)
#     y_pred_list.append(y_preds_random[0,:])
#     mse_list.append(mse_random)
#     t_list_list.append(t_list_random)
    
    
# plt.figure()
# for i in range(len(mse_list)):
#     plt.semilogy(t_list_list[i], mse_list[i], 'r', linewidth = 0.5)#, label = path_list[i])

# plt.semilogy(t_list_list[i], mse_list[i], 'r', linewidth = 0.5, label="random")


# for k in range(len(preds_mse)):
#     print(k)
#     err = preds_mse[k]
#     mean = err.mean(0).detach().numpy()
#     plt.semilogy(mean, label='$\Delta\ t$={}dt'.format(step_sizes[k]))
    
# plt.semilogy(multiscale_err, 'k', label = "multiscale")

# ts, means = find_ave_random_paths(t_list_list, mse_list)
# plt.plot(ts, means[:,0],  label = "average")


# plt.legend()
# plt.title(system + ": MSE : noise = " + str(noise) + ": " + str(num_lines) + " random paths, original perfect")
# plt.savefig("{}_MSE_n{}_all_original_perfect.pdf".format(system, noise))

# plt.figure()
# for i in range(len(y_pred_list)):
#     plt.plot(t_list_list[i], y_pred_list[i], 'r')
# plt.plot(test_data[0,1:], 'b')
# plt.title(system + ": Predicted 1 path : noise = " + str(noise) +" original perfect")
# plt.savefig("{}_predict_first_n{}_original_perfect.pdf".format(system, noise))

# plt.figure()
# for i in range(len(y_pred_list)):
#     plt.plot(t_list_list[i], y_pred_list[i], 'r')
# plt.plot(test_data[0,1:], 'b')
# plt.xlim([3000, test_data.shape[1]])
# plt.title(system + ": Predicted 1 path zoomed: noise = " + str(noise)+" original perfect")
# plt.savefig("{}_predict_first_zoomed_n{}_original_perfect.pdf".format(system, noise))

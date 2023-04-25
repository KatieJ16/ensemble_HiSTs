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
# from tqdm.notebook import tqdm

module_path = os.path.abspath(os.path.join('../../src/'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import ResNet as net_regular

    
import Resnet_multiscale_general as net


mse_list = list()
step_size_list = list()
noise_list = list()
t_list_all = list()

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
def predict_random_combo(model1, test_data, timesteps = 5000, to_plot=True):

    t = 0

    idx_combo = random.sample(range(len(all_combos)), 1)
    path = all_combos[idx_combo[0]]
    print(path)

    steps_per_combo = sum(path)

    path_loops = timesteps // steps_per_combo - 1

    n_timepoints = path_loops * len(path)

    n_test_points, _, ndim = test_data.shape


    t_list = np.zeros(n_timepoints)
    y_pred_list = np.zeros((n_test_points, n_timepoints, ndim))

    this_step_size = path[0]
    t+= this_step_size
    y_preds = model1.forward(torch.tensor(test_data[:, 0]).float(), str(this_step_size))
    y_pred_list[:,0] = y_preds.detach().numpy()
    t_list[0] = t
#     print(y_preds.shape)

    for j in range(n_timepoints-1):
        this_step_size = path[j%len(path)]
        t+= this_step_size
        y_preds = model1.forward(y_preds, str(this_step_size))

        y_pred_list[:,j+1] = y_preds.detach().numpy()
        t_list[j+1] = t

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

    return y_pred_list, mse, t_list, path
#===========================================================================================================
def predict_single_scale(model1, step_size, test_data, timesteps = 5000, size='small', to_plot=True):

    t = 0
    print(step_size)

    n_timepoints = timesteps // step_size + 1

    n_test_points, _, ndim = test_data.shape


    t_list = np.zeros(n_timepoints)
    y_pred_list = np.zeros((n_test_points, n_timepoints, ndim))

    t+= step_size
    y_preds = model1.forward(torch.tensor(test_data[:, 0]).float(), size)
    y_pred_list[:,0] = y_preds.detach().numpy()
    t_list[0] = t
    print(y_preds.shape)

    for j in range(n_timepoints-1):
        t+= step_size
        y_preds = model1.forward(y_preds, size)

        y_pred_list[:,j+1] = y_preds.detach().numpy()
        t_list[j+1] = t


    # need to interpolate between timesteps
    y_pred_list_new = np.zeros((n_test_points, timesteps - step_size, ndim))
    t_list_new = np.arange(step_size, timesteps)
    for i in range(len(y_pred_list)):
        for j in range(2):
            f = interp1d(t_list, y_pred_list[i,:,j])
            y_pred_list_new[i,:,j] = f(t_list_new)

    plt_idx = 0

    if to_plot:

        plt.plot(t_list_new, test_data[plt_idx,t_list_new.astype(int), 1])
        plt.plot(t_list_new, test_data[plt_idx,t_list_new.astype(int), 0])

        plt.plot(t_list_new, y_pred_list_new[plt_idx,:, 1])
        plt.plot(t_list_new, y_pred_list_new[plt_idx,:, 0])
        plt.title(system + ": step_size = " + str(step_size)+ ": noise = "+ str(noise))

        plt.ylim([np.min(test_data),np.max(test_data)])

        plt.show()

    mse = np.mean((y_pred_list_new - test_data[:,t_list_new.astype(int)])**2, axis = (0,2))
    if to_plot:
        plt.semilogy(t_list_new, mse)
        plt.title(system + ": step_size = " + str(step_size)+ ": noise = "+ str(noise))
        plt.show()


    return y_pred_list_new, mse, t_list_new

#===========================================================================================================
def vectorized_multi_scale_forecast(x_init, n_steps, models, step_sizes = [8,4]):
    """
    :param x_init: initial state torch array of shape n_test x n_dim
    :param n_steps: number of steps forward in terms of dt
    :param models: a list of models
    :return: a torch array of size n_test x n_steps x n_dim,
             a list of indices that are not achieved by interpolations
    """
    # sort models by their step sizes (decreasing order)
#         step_sizes = [model.step_size for model in models]
#         step_sizes = 
#         models = [model for _, model in sorted(zip(step_sizes, models), reverse=True)]



    # we assume models are sorted by their step sizes (decreasing order)
    n_test, n_dim = x_init.shape
    device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'
    indices = list()
    extended_n_steps = n_steps + max(models.step_sizes) #models[0].step_size
    preds = torch.zeros(n_test, extended_n_steps + 1, n_dim).float().to(device)

    # vectorized simulation
    indices.append(0)
    preds[:, 0, :] = x_init
    total_step_sizes = n_steps
#         for model in models:
#         type_models = [16, 8, 4]#['large', 'small']
#         for i in [0, 1, 2]:
#         step_sizes = [216, 8, 4]
    for step_size in step_sizes:
#             step_size = step_sizes[i]
        type_model = str(step_size)#type_models[i]
        n_forward = int(total_step_sizes/step_size)
        y_prev = preds[:, indices, :].reshape(-1, n_dim)
        indices_lists = [indices]
        for t in range(n_forward):
            y_next = models.forward(y_prev, str(type_model))
            shifted_indices = [x + (t + 1) * step_size for x in indices]
            indices_lists.append(shifted_indices)
            preds[:, shifted_indices, :] = y_next.reshape(n_test, -1, n_dim)
            y_prev = y_next
        indices = [val for tup in zip(*indices_lists) for val in tup]
        total_step_sizes = step_size - 1

    # simulate the tails
    last_idx = indices[-1]
    y_prev = preds[:, last_idx, :]
    while last_idx < n_steps:
        last_idx += step_sizes[-1]
        type_model = str(step_sizes[-1])#type_models[-1]
        y_next = models.forward(y_prev, type_model)
        preds[:, last_idx, :] = y_next
        indices.append(last_idx)
        y_prev = y_next

    # interpolations
    sample_steps = range(1, n_steps+1)
    valid_preds = preds[:, indices, :].detach().numpy()
    cs = scipy.interpolate.interp1d(indices, valid_preds, kind='linear', axis=1)
    y_preds = torch.tensor(cs(sample_steps)).float()

    return y_preds
#===========================================================================================================
def find_ave_random_paths(t_list_list, mse_list):
    to_ave = np.zeros((num_lines, 10000, test_data.shape[-1]))
    print(to_ave.shape)
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

def predict_random_combo_not_repeat(models, test_data, timesteps = 5000, to_plot=True):


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
        y_preds = models.forward(y_preds, str(this_step_size))

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
#===========================================================================================================

# adjustables

# system = 'VanDerPol'         # system name: 'Hyperbolic', 'Cubic', 'VanDerPol', 'Hopf' or 'Lorenz'
# system = 'Lorenz'

# for noise in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]:
#     noise = 0.1                  # noise percentage: 0.00, 0.01 or 0.02
#     letter = 'a'

step_sizes = [4, 8, 216]#[4, 8, 216]
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
    step_sizes = [16, 32, 64]
    
elif system == "hyperbolic":
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
    step_sizes = [smallest_step, smallest_step*2, smallest_step*4]
    
n_poss = 1773


# paths
data_dir = os.path.join('../../data/', system,)
model_dir = os.path.join('../../models/', system)


# load data
#     noise = 0.0
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

# perfect_data = np.load(os.path.join(data_dir, 'test_noise{}.npy'.format(0.0)))

# n_steps = train_data.shape[1]

n_train = train_data.shape[0]
n_val = val_data.shape[0]
n_test = test_data.shape[0]

print("train_data shape = ", train_data.shape)
print("val_data shape = ", val_data.shape)
print("test_data.shape = ", test_data.shape)

    #make and train
# model_name = 'model_{}_depends{}_noise{}_{}.pt'.format(system, n_poss, noise, letter)
model_name = 'model_{}_depends_stepsize{}_noise{}_{}.pt'.format(system, smallest_step, noise, letter)
# model_name = 'model_{}_depends{}_stepsize{}_noise{}_{}.pt'.format(system, n_poss, smallest_step, noise, letter)
print("train_data shape = ", train_data.shape)
print("val_data shape = ", val_data.shape)
print("test_data.shape = ", test_data.shape)


model = torch.load(os.path.join(model_dir, model_name), map_location='cpu')
model.device = 'cpu'

print("model.step_sizes= ", model.step_sizes)
for step_size in model.step_sizes:
    model._modules[str(step_size)]._modules['activation'] = torch.nn.ReLU()

    

to_plot = False
timesteps = test_data.shape[1]
print("timesteps = ", timesteps)

#get each single stepper
y_pred_list_single = list()
mse_list_single = list()
t_list_single = list()

for step_size in step_sizes:

    y_pred_list, mse, t_list = predict_single_scale(model, step_size, test_data, timesteps = timesteps, 
                                                            size=str(step_size), to_plot=to_plot)
    y_pred_list_single.append(y_pred_list)
    mse_list_single.append(mse)
    t_list_single.append(t_list)

#get multiscale stepper   
n_steps = 5000
y_preds = vectorized_multi_scale_forecast(torch.tensor(test_data[:, 0, :]).float(), n_steps, model, step_sizes = step_sizes[::-1])

all_combos = np.load('all_combos_'+str(smallest_step)+'.npy', allow_pickle=True)


num_lines = 500
mse_list = list()
t_list_list = list()
path_list = list()
y_pred_list = list()

for i in range(num_lines):
    print("i = ", i)
#     y_preds_random, mse_random, t_list_random, path_random = predict_random_combo(model, test_data, to_plot = False,  timesteps = 5000)
    y_preds_random, mse_random, t_list_random = predict_random_combo_not_repeat(model, test_data, timesteps = 5000, to_plot=False)
    y_pred_list.append(y_preds_random[0,:])
    mse_list.append(mse_random)
    t_list_list.append(t_list_random)
#     path_list.append(path_random)

ts, means, stds = find_ave_random_paths(t_list_list, mse_list)

plt.figure()
for i in range(len(mse_list)):
    plt.semilogy(t_list_list[i], mse_list[i], 'r', linewidth = 0.5)#, label = path_list[i])

plt.semilogy(t_list_list[i], mse_list[i], 'r', linewidth = 0.5, label="random")

#plot each one
labels = ["small", "mid", "large"]
for i in range(len(y_pred_list_single)):
    plt.semilogy(t_list_single[i], mse_list_single[i],  label = labels[i])

mse = torch.mean((y_preds - torch.tensor(test_data[:, 1:1+n_steps, :]).float())**2, axis = (0,2))
plt.semilogy(mse, 'k', label = "multiscale")
plt.semilogy(ts, means[:,0], label="average")
plt.legend()
plt.title(system + ": MSE : noise = " + str(noise) + ": " + str(num_lines) + " random paths :step_sizes = "+str(step_sizes))
plt.savefig("{}_{}_MSE_n{}_all.pdf".format(system, letter, noise))

plt.figure()
for i in range(len(y_pred_list)):
    plt.plot(t_list_list[i], y_pred_list[i], 'r')
plt.plot(test_data[0,1:], 'b')
plt.title(system + ": Predicted 1 path : noise = " + str(noise) +" :step_sizes = "+str(step_sizes))
plt.savefig("{}_{}_predict_first_n{}.pdf".format(system, letter, noise))

# plt.figure()
# for i in range(len(y_pred_list)):
#     plt.plot(t_list_list[i], y_pred_list[i], 'r')
# plt.plot(test_data[0,1:], 'b')
# plt.xlim([3000, test_data.shape[1]])
# plt.title(system + ": Predicted 1 path zoomed: noise = " + str(noise) +" :step_sizes = "+str(step_sizes))
# plt.savefig("{}_{}_predict_first_zoomed_n{}.pdf".format(system, letter, noise))


#want a plot with of the first test plot. Where the predicted has average of all paths in thick line and shaded for 1 and 2 stds. 

n_timesteps = 1e10
for i in range(len(t_list_list)):
    if n_timesteps > max(t_list_list[i]):
        n_timesteps = max(t_list_list[i])
# n_timesteps = np.min(np.max(t_list_list[i], 1))
print("n_timesteps =", n_timesteps)
y_preds_all = np.zeros((len(y_pred_list), n_timesteps, train_data.shape[2]))
print("y_preds_all shape = ", y_preds_all.shape)
#we need to interpolate y_preds
for i in range(len(y_pred_list)):
#     preds = torch.stack(preds, 2).detach().numpy()
    #insert first step
#     print("np.insert(t_list_list[i], 0, 0) = ", np.insert(t_list_list[i], 0, 0))
#     print("y_pred_list ", y_pred_list[i].shape)
    t = np.insert(t_list_list[i], 0, 0)
    y = np.insert(y_pred_list[i], 0, test_data[0,0], axis = 0)
#     print("y shape = ", y.shape)
#     y = y[:, np.newaxis]
    
#     print("t_list_list[i] = ", t)
    sample_steps = range(0, n_timesteps)
#     print("t_list_list[i] shape = ", t.shape)
#     print("y_pred_list[i] shape = ", y.shape)
    cs = scipy.interpolate.interp1d(t, y.T, kind='linear')
    y_preds = cs(sample_steps).T#torch.tensor(cs(sample_steps)).transpose(1, 2).float()
        
#     print("y_preds shape = ", y_preds.shape)
    y_preds_all[i] = y_preds

#     plt.figure()
#     plt.plot(t, y, label = "original")
#     plt.plot(sample_steps, y_preds, label = "intepolated")
#     plt.legend()
#     plt.savefig("interpolate.pdf")
#find average solution 

# to_ave = np.zeros((num_lines, 10000, test_data.shape[-1]))
# print('to_ave = ', to_ave.shape)
# print("y_pred_list len = ", len(y_pred_list))
# print("y_pred_list[0] shape = ", y_pred_list[0].shape)
# for i in range(num_lines):
#     for j in range(len(t_list_list[i])):
#         to_ave[i, int(t_list_list[i][j])] = y_pred_list[i][j]

# averages = np.zeros(10000)
# to_ave[to_ave == 0] = np.nan
# means = np.nanmean(to_ave[:, 1:], axis=0)
# stds = np.nanstd(to_ave[:, 1:], axis=0)

# mask = np.isfinite(means[:,0])
# ts = np.arange(1, 10000)

# ts = ts[mask]
# means = means[mask]
# stds = stds[mask]

means = np.mean(y_preds_all, axis = 0)
stds = np.std(y_preds_all, axis = 0)
ts = sample_steps

# print("ts.shape = ", ts.shape)
print("means.shape = ", means.shape)
print("stds.shape = ", stds.shape)

plt.figure()
plt.plot(test_data[0,1:], 'b')
plt.plot(ts, means, 'r')
for i in range(len(means[0])):
    for n_stds in range(1, 4):
        plt.fill_between(ts, means[:,i]+n_stds*stds[:,i], means[:,i]-n_stds*stds[:,i], facecolor='red', alpha=0.25)
        

plt.title(system + ": Predicted 1 path: noise = " + str(noise) +" :step_sizes = "+str(step_sizes))
plt.savefig("{}_{}_predict_first_shaded_n{}.pdf".format(system, letter, noise))


#plot the training and validation errors
plt.figure()
plt.semilogy(model.train_errors, label = "train")
plt.semilogy(model.val_errors, label = "validation")
plt.legend()

plt.savefig("errors.pdf")

# #===========================================================================================================
# #repeat with the perfect data
# to_plot = False
# timesteps = perfect_data.shape[1]

# #get each single stepper
# y_pred_list_single = list()
# mse_list_single = list()
# t_list_single = list()

# for step_size in step_sizes:

#     y_pred_list, mse, t_list = predict_single_scale(model, step_size, perfect_data, timesteps = timesteps, 
#                                                             size=str(step_size), to_plot=to_plot)
#     y_pred_list_single.append(y_pred_list)
#     mse_list_single.append(mse)
#     t_list_single.append(t_list)

# #get multiscale stepper   
# n_steps = 5000
# y_preds = vectorized_multi_scale_forecast(torch.tensor(perfect_data[:, 0, :]).float(), n_steps, model, step_sizes = step_sizes[::-1])

# all_combos = np.load('all_combos_'+str(smallest_step)+'.npy', allow_pickle=True)


# num_lines = 500
# mse_list = list()
# t_list_list = list()
# path_list = list()
# y_pred_list = list()

# for i in range(num_lines):
#     print("i = ", i)
#     y_preds_random, mse_random, t_list_random, path_random = predict_random_combo(model, perfect_data, to_plot = False,  timesteps = 5000)
#     y_pred_list.append(y_preds_random[0,:])
#     mse_list.append(mse_random)
#     t_list_list.append(t_list_random)
#     path_list.append(path_random)

# ts, means, stds = find_ave_random_paths(t_list_list, mse_list)

# plt.figure()
# for i in range(len(mse_list)):
#     plt.semilogy(t_list_list[i], mse_list[i], 'r', linewidth = 0.5)#, label = path_list[i])

# plt.semilogy(t_list_list[i], mse_list[i], 'r', linewidth = 0.5, label="random")

# #plot each one
# labels = ["small", "mid", "large"]
# for i in range(len(y_pred_list_single)):
#     plt.semilogy(t_list_single[i], mse_list_single[i],  label = labels[i])

# mse = torch.mean((y_preds - torch.tensor(perfect_data[:, 1:1+n_steps, :]).float())**2, axis = (0,2))
# plt.semilogy(mse, 'k', label = "multiscale")
# plt.semilogy(ts, means[:,0], label="average")
# plt.legend()
# plt.title(system + ": MSE : noise = " + str(noise) + ": " + str(num_lines) + " random paths, with perfect data")
# plt.savefig("{}_MSE_n{}_all_perfect.pdf".format(system, noise))

# plt.figure()
# for i in range(len(y_pred_list)):
#     plt.plot(t_list_list[i], y_pred_list[i], 'r')
# plt.plot(perfect_data[0,1:], 'b')
# plt.title(system + ": Predicted 1 path : noise = " + str(noise) +", with perfect data" )
# plt.savefig("{}_predict_first_n{}_perfect.pdf".format(system, noise))

# plt.figure()
# for i in range(len(y_pred_list)):
#     plt.plot(t_list_list[i], y_pred_list[i], 'r')
# plt.plot(perfect_data[0,1:], 'b')
# plt.xlim([3000, perfect_data.shape[1]])
# plt.title(system + ": Predicted 1 path zoomed: noise = " + str(noise)+", with perfect data" )
# plt.savefig("{}_predict_first_zoomed_n{}_perfect.pdf".format(system, noise))

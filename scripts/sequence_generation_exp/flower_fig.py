
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

system = 'flower'
data_dir = os.path.join('../../data/', system,)
model_dir = os.path.join('../../models/', system)

t_list_list = np.load(os.path.join(model_dir, 't_list_list.npy'), allow_pickle=True)
mse_list_depends = np.load(os.path.join(model_dir, 'mse_list_depends.npy'), allow_pickle=True)
mse_list_original = np.load(os.path.join(model_dir, 'mse_list_original.npy'), allow_pickle=True)
y_pred_list_depends = np.load(os.path.join(model_dir, 'y_pred_list_depends.npy'), allow_pickle=True)
y_pred_list_original = np.load(os.path.join(model_dir, 'y_pred_list_original.npy'), allow_pickle=True)

ts_depends = np.load(os.path.join(model_dir, 'ts_depends.npy'))
means_depends = np.load(os.path.join(model_dir, 'means_depends.npy'))
stds_depends = np.load(os.path.join(model_dir, 'stds_depends.npy'))
percent_95_depends = np.load(os.path.join(model_dir, 'percent_95_depends.npy'))
percent_5_depends = np.load(os.path.join(model_dir, 'percent_5_depends.npy'))
ts_original = np.load(os.path.join(model_dir, 'ts_original.npy'))
means_original = np.load(os.path.join(model_dir, 'means_original.npy'))
stds_original = np.load(os.path.join(model_dir, 'stds_original.npy'))
percent_95_original = np.load(os.path.join(model_dir, 'percent_95_original.npy'))
percent_5_original = np.load(os.path.join(model_dir, 'percent_5_original.npy'))


num_lines = len(t_list_list)

U = np.load(os.path.join(data_dir, 'U.npy'))
s = np.load(os.path.join(data_dir, 's.npy'))


fig = plt.figure(figsize=(12, 3))
# plt.plot(coupled_nn_data_true[120:180, 220:280, :].mean(0).mean(0).mean(0), 'g.')
for i in range(100):#len(y_pred_list_depends)):
    print(i)
#     print(len(y_pred_list_depends[i]))
    coupled_nn_V = y_pred_list_original[i]
    coupled_nn_data = (U[:, :64]@np.diag(s[:64])@coupled_nn_V.T).reshape(540, 960, 3, -1)
#     fig = plt.figure(figsize=(12, 3))
#     print(coupled_nn_data[100:110, 70:80, :].mean(0).mean(0).shape)
    plt.plot(t_list_list[i], coupled_nn_data[120:180, 220:280, :].mean(0).mean(0).mean(0), 'b.')
#     print(coupled_nn_data[100:110, 70:80, :].mean(0).mean(0).shape)
plt.xticks([], [])
plt.yticks([], [])
plt.ylim([0,255])
plt.savefig('flower_fig_original.png')
"""
    The functions used for post processing with the random paths
"""
# import os
# import sys
import torch
import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d
# import time
# import scipy
# import random

# import Resnet_multiscale_general as net

#===========================================================================================================
def find_ave_random_paths(t_list_list, mse_list, test_data, num_lines = 100, ):
    """
        Finds the average mse of randon paths
    """
    to_ave = np.zeros((num_lines, 10000, test_data.shape[-1]))
    for i in range(num_lines):
        for j in range(len(t_list_list[i])):
            to_ave[i, int(t_list_list[i][j])] = mse_list[i][j]

    to_ave[to_ave == 0] = np.nan
    means = np.nanmean(to_ave[:, 1:], axis=0)
    stds = np.nanstd(to_ave[:, 1:], axis=0)
    percent_95 = np.nanpercentile(to_ave[:, 1:], 95, axis=0)
    percent_5 = np.nanpercentile(to_ave[:, 1:], 5, axis=0)

    mask = np.isfinite(means[:, 0])
    ts = np.arange(1, 10000)

    return ts[mask], means[mask], stds[mask], percent_95[mask], percent_5[mask]

#===========================================================================================================

def predict_random_combo(model_depends, models_original, test_data, timesteps=5000, step_sizes=[1, 2, 4]):
    """
        Predicts 1 random path of both the depends and original models.
        Uses the same paths to predict on all
    """

    t = 0
    n_test_points, _, ndim = test_data.shape


    t_list = list()
    y_pred_list_depends = list()
    y_pred_list_original = list()

    indices = np.random.randint(0, len(step_sizes), int(timesteps/min(step_sizes)))
    steps = list()
    for i in range(len(indices)):
        steps.append(step_sizes[indices[i]])
        if sum(steps) > timesteps:
            break
    
    y_pred_list_depends = np.zeros((n_test_points, len(steps)-1, ndim))
    y_pred_list_original = np.zeros((n_test_points, len(steps)-1, ndim))
    y_preds_depends = torch.tensor(test_data[:, 0]).float()
    y_preds_original = torch.tensor(test_data[:, 0]).float()

    for i in range(len(steps)-1):
        this_pick = indices[i]
        this_step_size = steps[i]
        t += this_step_size
        y_preds_depends = model_depends.forward(y_preds_depends, str(this_step_size))
        y_preds_original = models_original[this_pick].forward(y_preds_original)

        y_pred_list_depends[:, i] = y_preds_depends.detach().numpy()
        y_pred_list_original[:, i] = y_preds_original.detach().numpy()
        t_list.append(t)
        
    t_list = np.array(t_list)
    y_pred_list_depends = np.array(y_pred_list_depends)
    y_pred_list_original = np.array(y_pred_list_original)
    
    
    mse_depends = np.mean((y_pred_list_depends - test_data[:, t_list.astype(int)])**2, axis=(0, 2))
    mse_original = np.mean((y_pred_list_original - test_data[:, t_list.astype(int)])**2, axis=(0, 2))
    
    return y_pred_list_depends, y_pred_list_original, mse_depends, mse_original, t_list
#===========================================================================================================

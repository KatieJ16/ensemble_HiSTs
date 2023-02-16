import torch
import random
import numpy as np


class DataSet:
    def __init__(self, train_data, val_data, test_data, dt, step_size, n_forward, backward = False):
        """
        :param train_data: array of shape n_train x train_steps x input_dim
                            where train_steps = max_step x (n_steps + 1)
        :param val_data: array of shape n_val x val_steps x input_dim
        :param test_data: array of shape n_test x test_steps x input_dim
        :param dt: the unit time step
        :param step_size: an integer indicating the step sizes
        :param n_forward: number of steps forward
        """
        n_train, train_steps, n_dim = train_data.shape
        n_val, val_steps, _ = val_data.shape
        n_test, test_steps, _ = test_data.shape
        assert step_size*n_forward+1 <= train_steps and step_size*n_forward+1 <= val_steps

        # params
        self.dt = dt
        self.n_dim = n_dim
        self.step_size = step_size
        self.n_forward = n_forward
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test

        # device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if backward:
            train_data = np.flip(train_data,axis=1).copy()
            val_data = np.flip(val_data,axis=1).copy()
            test_data = np.flip(test_data,axis=1).copy()
        # data
#         if not backward:
        x_idx = 0
        y_start_idx = x_idx + step_size
        y_end_idx = x_idx + step_size*n_forward + 1
        self.train_x = torch.tensor(train_data[:, x_idx, :]).float().to(self.device)
        self.train_ys = torch.tensor(train_data[:, y_start_idx:y_end_idx:step_size, :]).float().to(self.device)
        self.val_x = torch.tensor(val_data[:, x_idx, :]).float().to(self.device)
        self.val_ys = torch.tensor(val_data[:, y_start_idx:y_end_idx:step_size, :]).float().to(self.device)
        self.test_x = torch.tensor(test_data[:, 0, :]).float().to(self.device)
        self.test_ys = torch.tensor(test_data[:, 1:, :]).float().to(self.device)

#         if backward:
#             # data
#             x_idx = -1
#             step_size = -1 * step_size
#             y_start_idx = x_idx + step_size
#             y_end_idx = x_idx + step_size*n_forward - 1
#             self.train_x = torch.tensor(train_data[:, x_idx, :]).float().to(self.device)
#             self.train_ys = torch.tensor(train_data[:, y_start_idx:y_end_idx:step_size, :]).float().to(self.device)
#             self.val_x = torch.tensor(val_data[:, x_idx, :]).float().to(self.device)
#             self.val_ys = torch.tensor(val_data[:, y_start_idx:y_end_idx:step_size, :]).float().to(self.device)
#             self.test_x = torch.tensor(test_data[:, 0, :]).float().to(self.device)
#             self.test_ys = torch.tensor(test_data[:, 1:, :]).float().to(self.device)

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
# from tqdm.notebook import tqdm

module_path = os.path.abspath(os.path.join('../../src/'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import ResNet as net

#===========================================================================================================
class DataSet:
    def __init__(self, train_data, val_data, test_data, dt, step_size, n_forward):
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
        self.device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'

        # data
        x_idx = 0
        y_start_idx = x_idx +1#+ step_size
        y_end_idx = x_idx +n_forward+1#+ step_size*n_forward + 1
        self.train_x = torch.tensor(train_data[:, x_idx, :], requires_grad=True).float().to(self.device)
        self.train_ys = torch.tensor(train_data[:, y_start_idx:y_end_idx:, :]).float().to(self.device)
        self.val_x = torch.tensor(val_data[:, x_idx, :], requires_grad=True).float().to(self.device)
        self.val_ys = torch.tensor(val_data[:, y_start_idx:y_end_idx:, :]).float().to(self.device)
        self.test_x = torch.tensor(test_data[:, 0, :], requires_grad=True).float().to(self.device)
        self.test_ys = torch.tensor(test_data[:, 1:, :]).float().to(self.device)
#===========================================================================================================

#Resnet.py 


class NNBlock(torch.nn.Module):
    def __init__(self, arch, activation=torch.nn.ReLU()):
        """
        :param arch: architecture of the nn_block
        :param activation: activation function
        """
        super(NNBlock, self).__init__()

        # param
        self.n_layers = len(arch)-1
        self.activation = activation
        self.device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'

        # network arch
        for i in range(self.n_layers):
            self.add_module('Linear_{}'.format(i), torch.nn.Linear(arch[i], arch[i+1]).to(self.device))

    def forward(self, x):
        """
        :param x: input of nn
        :return: output of nn
        """
        for i in range(self.n_layers - 1):
            x = self.activation(self._modules['Linear_{}'.format(i)](x))
        # no nonlinear activations in the last layer
        x = self._modules['Linear_{}'.format(self.n_layers - 1)](x)
        return x


class ResNet(torch.nn.Module):
    def __init__(self, arch, dt, step_size, prev_models, activation=torch.nn.ReLU()):
        """
        :param arch: a list that provides the architecture
        :param dt: time step unit
        :param step_size: forward step size
        :param activation: activation function in neural network
        """
        super(ResNet, self).__init__()

        # check consistencies
        assert isinstance(arch, list)
        assert arch[0] == arch[-1]

        # param
        self.n_dim = arch[0]

        # data
        self.dt = dt
        self.step_size = step_size

        # device
        self.device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'

        # layer
        self.activation = activation
        self.add_module('increment', NNBlock(arch, activation=activation))
        
        # sort models by their step sizes (increasing order)
        step_sizes = [model.step_size for model in prev_models]
        models = [model for _, model in sorted(zip(step_sizes, prev_models), reverse=False)]
    
        self.prev_models = models
        
    def check_data_info(self, dataset):
        """
        :param: dataset: a dataset object
        :return: None
        """
        assert self.n_dim == dataset.n_dim
        assert self.dt == dataset.dt
        assert self.step_size == dataset.step_size

    def forward(self, x_init):
        """
        :param x_init: array of shape batch_size x input_dim
        :return: next step prediction of shape batch_size x input_dim
        """
        return x_init + self._modules['increment'](x_init)

    def uni_scale_forecast(self, x_init, n_steps):
        """
        :param x_init: array of shape n_test x input_dim
        :param n_steps: number of steps forward in terms of dt
        :return: predictions of shape n_test x n_steps x input_dim and the steps
        """
        steps = list()
        preds = list()
        sample_steps = range(n_steps)

        # forward predictions
        x_prev = x_init
        cur_step = self.step_size - 1
        while cur_step < n_steps + self.step_size:
            x_next = self.forward(x_prev)
            steps.append(cur_step)
            preds.append(x_next)
            cur_step += self.step_size
            x_prev = x_next

        # include the initial frame
        steps.insert(0, 0)
        preds.insert(0, torch.tensor(x_init).float().to(self.device))

        # interpolations
        preds = torch.stack(preds, 2).detach().numpy()
        cs = scipy.interpolate.interp1d(steps, preds, kind='linear')
        y_preds = torch.tensor(cs(sample_steps)).transpose(1, 2).float()

        return y_preds

    def train_net(self, dataset, max_epoch, batch_size, w=1.0, lr=1e-3, model_path=None):
        """
        :param dataset: a dataset object
        :param max_epoch: maximum number of epochs
        :param batch_size: batch size
        :param w: l2 error weight
        :param lr: learning rate
        :param model_path: path to save the model
        :return: None
        """
        # check consistency
        self.check_data_info(dataset)

        # training
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        epoch = 0
        best_loss = 1e+5
        start_time = time.time()
        while epoch < max_epoch:
            epoch += 1
            # ================= prepare data ==================
            n_samples = dataset.n_train
            new_idxs = torch.randperm(n_samples)
            batch_x = dataset.train_x[new_idxs[:batch_size], :].to(self.device)
            batch_ys = dataset.train_ys[new_idxs[:batch_size], :, :].to(self.device)
            # =============== calculate losses ================
            train_loss = self.calculate_loss(batch_x, batch_ys, w=w)
            val_loss = self.calculate_loss(dataset.val_x, dataset.val_ys, w=w)
            # ================ early stopping =================
            if best_loss <= 1e-8:
                print('--> model has reached an accuracy of 1e-8! Finished training!')
                break
            # =================== backward ====================
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            # =================== log =========================
            
            if epoch == 10:
                end_time = time.time()
                print("time for first 10 = ", end_time - start_time)
            if epoch % 10 == 0:
                print('epoch {}, training loss {}, validation loss {}'.format(epoch, train_loss.item(),
                                                                              val_loss.item()))
                if val_loss.item() < best_loss:
                    best_loss = val_loss.item()
                    if model_path is not None:
                        print('(--> new model saved @ epoch {})'.format(epoch))
                        torch.save(self, model_path)

        # if to save at the end
        if val_loss.item() < best_loss and model_path is not None:
            print('--> new model saved @ epoch {}'.format(epoch))
            torch.save(self, model_path)

    def calculate_loss(self, x, ys, w=1.0):
        """
        :param x: x batch, array of size batch_size x n_dim
        :param ys: ys batch, array of size batch_size x n_steps x n_dim
        :return: overall loss
        """
        torch.autograd.set_detect_anomaly(True)
        batch_size, n_steps, n_dim = ys.size()
        assert n_dim == self.n_dim
        
        #we want to get next 5 timesteps [1 small, 1big, 1big, 1small, 2big]
        y_preds = torch.zeros(batch_size, n_steps, n_dim).float().to(self.device)
        
        
        #first a small
        y_preds[:,0,:] = self.prev_models[0](x)
        
        preds_big = self.forward(x)
        #next is big
        y_preds[:,1,:] = preds_big
        #big then small 
        y_preds[:,2,:] = self.prev_models[0](preds_big)
        #two big 
        preds_big = self.forward(preds_big)
        y_preds[:,3,:] = preds_big
        #2 big and a small 
        y_preds[:,4,:] = self.prev_models[0](preds_big)
        

        # compute loss
        criterion = torch.nn.MSELoss(reduction='none')
        loss = w * criterion(y_preds, ys).mean() + (1-w) * criterion(y_preds, ys).max()
        


        return loss



def vectorized_multi_scale_forecast(x_init, n_steps, models):
    """
    :param x_init: initial state torch array of shape n_test x n_dim
    :param n_steps: number of steps forward in terms of dt
    :param models: a list of models
    :return: a torch array of size n_test x n_steps x n_dim,
             a list of indices that are not achieved by interpolations
    """
    
    # sort models by their step sizes (decreasing order)
    step_sizes = [model.step_size for model in models]
    models = [model for _, model in sorted(zip(step_sizes, models), reverse=True)]

    # we assume models are sorted by their step sizes (decreasing order)
    n_test, n_dim = x_init.shape
    device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'
    indices = list()
    extended_n_steps = n_steps + models[0].step_size
    preds = torch.zeros(n_test, extended_n_steps + 1, n_dim,requires_grad=True).float().to(device)

    # vectorized simulation
    indices.append(0)
    print(x_init.requires_grad)
    print(preds.requires_grad)
    preds[:, 0, :] = x_init
    print(preds.requires_grad)
    total_step_sizes = n_steps
    for model in models:
        n_forward = int(total_step_sizes/model.step_size)
        y_prev = preds[:, indices, :].reshape(-1, n_dim)
        indices_lists = [indices]
        for t in range(n_forward):
            y_next = model(y_prev.to(device)).to(device)
            shifted_indices = [x + (t + 1) * model.step_size for x in indices]
            indices_lists.append(shifted_indices)
            preds[:, shifted_indices, :] = y_next.reshape(n_test, -1, n_dim)
            y_prev = y_next
        indices = [val for tup in zip(*indices_lists) for val in tup]
        total_step_sizes = model.step_size - 1

    # simulate the tails
    last_idx = indices[-1]
    y_prev = preds[:, last_idx, :]
    while last_idx < n_steps:
        last_idx += models[-1].step_size
        y_next = models[-1](y_prev)
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

# adjustables

k = 1                        # model index: should be in {0, 2, ..., 10}
dt = 0.01                     # time unit: 0.0005 for Lorenz and 0.01 for others
system = 'Hyperbolic'         # system name: 'Hyperbolic', 'Cubic', 'VanDerPol', 'Hopf' or 'Lorenz'
noise = 0.0                   # noise percentage: 0.00, 0.01 or 0.02

lr = 1e-3                     # learning rate
max_epoch = 100000            # the maximum training epoch 
batch_size = 320              # training batch size
arch = [2, 128, 128, 128, 2]  # architecture of the neural network

# paths
data_dir = os.path.join('../../data/', system)
model_dir = os.path.join('../../models/', system)

# global const
n_forward = 5
step_size = 2**k



        
# load data
train_data = np.load(os.path.join(data_dir, 'train_noise{}.npy'.format(noise)))
val_data = np.load(os.path.join(data_dir, 'val_noise{}.npy'.format(noise)))
test_data = np.load(os.path.join(data_dir, 'test_noise{}.npy'.format(noise)))
n_train = train_data.shape[0]
n_val = val_data.shape[0]
n_test = test_data.shape[0]

# create dataset object
dataset = DataSet(train_data, val_data, test_data, dt, step_size, n_forward)

prev_step_sizes = [1]
prev_models = list()
for s in prev_step_sizes:
    print('load model_D{}.pt'.format(s))
    prev_models.append(torch.load(os.path.join(model_dir, 'model_D{}_noise0.0.pt'.format(s)), map_location='cpu'))
    
# fix model consistencies trained on gpus (optional)
for model in prev_models:
    model.device = 'cpu'
    model._modules['increment']._modules['activation'] = torch.nn.ReLU()
    
#make and train
model_name = 'model_D{}_noise{}_depends.pt'.format(step_size, noise)

# create/load model object
print('create model {} ...'.format(model_name))
model = ResNet(arch=arch, dt=dt, step_size=step_size, prev_models=prev_models)

# training
model.train_net(dataset, max_epoch=max_epoch, batch_size=batch_size, lr=lr,
                model_path=os.path.join(model_dir, model_name))
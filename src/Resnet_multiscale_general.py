
import time

import numpy as np
import torch
import scipy.interpolate


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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    def __init__(self, arch, dt, step_sizes, activation=torch.nn.ReLU(), n_poss=25, combos_file=None):
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
        self.n_poss = n_poss

        # data
        self.dt = dt
        self.step_sizes = step_sizes

        # device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.train_errors = list()
        self.val_errors = list()

        # layer
        self.activation = activation
        for step_size in step_sizes:
            self.add_module(str(step_size), NNBlock(arch, activation=activation))
            
        try: #first try to load file
            if combos_file is None:
                combos_file = 'all_combos_'+str(min(step_sizes))+'.npy'
            self.all_combos = np.load(combos_file, allow_pickle=True).flatten()

            print("Things about all combos:")
            print("len all_combos = ", len(self.all_combos))
            print("first 10  = ", self.all_combos[:10])
            print("last 10 = ", self.all_combos[-10:])
            
        except: #or make it
            print("Couldn't Load combos, making")
#             sys.exit(1)
            self.all_combos = self.make_all_combos(file_name='all_combos_'+str(min(step_sizes))+'.npy')
            #save to use next time. 
            print("saving for later")
            print('all_combos_'+str(min(step_sizes))+'.npy')
            np.save('all_combos_'+str(min(step_sizes))+'.npy', self.all_combos, allow_pickle=True)
            
        self.best_loss = 1e+5
        

    def check_data_info(self, dataset):
        """
        :param: dataset: a dataset object
        :return: None
        """
        return
        assert self.n_dim == dataset.n_dim
        assert self.dt == dataset.dt
        assert self.step_size == dataset.step_size

    def forward(self, x_init, size):
        """
        :param x_init: array of shape batch_size x input_dim
        :param type: type of step size, small, mid, large
        :return: next step prediction of shape batch_size x input_dim
        """
        return x_init + self._modules[size](x_init)

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

    def train_net(self, dataset, max_epoch, batch_size, w=1.0, lr=1e-3, model_path=None, print_every=1000):
        """
        :param dataset: a dataset object
        :param max_epoch: maximum number of epochs
        :param batch_size: batch size
        :param w: l2 error weight
        :param lr: learning rate
        :param model_path: path to save the model
        :return: None
        """
        
        print("len(all_combos) = ", len(self.all_combos))
        
        # check consistency
        self.check_data_info(dataset)

        # training
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        epoch = 0
        start_time = time.time()
        while epoch < max_epoch:
            epoch += 1
            # ================= prepare data ==================
            n_samples = dataset.n_train
            new_idxs = torch.randperm(n_samples)
            batch_x = dataset.train_x[new_idxs[:batch_size], :]
#             print("dataset.train_ys shape = ", dataset.train_ys.shape)
            batch_ys = dataset.train_ys[new_idxs[:batch_size], :, :]
            # =============== calculate losses ================
#             print("batch_ys.shape = ", batch_ys.shape)
            train_loss = self.calculate_loss(batch_x, batch_ys, w=w)
#             val_loss = self.calculate_loss(dataset.val_x, dataset.val_ys, w=w, limit=False)
            
            # =================== backward ====================
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            # =================== log =========================
            
            if epoch == 10:
                end_time = time.time()
                print("time for first 10 = ", end_time - start_time)
            
            if epoch % print_every == 0:
                val_loss = self.calculate_loss(dataset.val_x, dataset.val_ys, w=w, limit=False)
                print('epoch {}, training loss {}, validation loss {}, best_loss {}'.format(epoch, train_loss.item(),
                                                                                            val_loss.item(), self.best_loss))
                self.train_errors.append(train_loss.item())
                self.val_errors.append(val_loss.item())
        
                if val_loss.item() < self.best_loss:
                    self.best_loss = val_loss.item()
                    if model_path is not None:
                        print('(--> new model saved @ epoch {})'.format(epoch))
#                         print('epoch {}, training loss {}, validation loss {}'.format(epoch, train_loss.item(),
#                                                                                   val_loss.item()))
#                         print("model_path = ", model_path)
                        torch.save(self, model_path)
    
            # ================ early stopping =================
            if self.best_loss <= 1e-8:
                print('--> model has reached an accuracy of 1e-8! Finished training!')
                break


        # if to save at the end
        if val_loss.item() < self.best_loss and model_path is not None:
            print('--> new model saved @ epoch {}'.format(epoch))
            torch.save(self, model_path)


    def calculate_loss(self, x, ys, w=1.0, limit=True):
        """
        :param x: x batch, array of size batch_size x n_dim
        :param ys: ys batch, array of size batch_size x n_steps x n_dim
        :return: overall loss
        """
        batch_size, n_steps, n_dim = ys.size()
        assert n_dim == self.n_dim
    
        criterion = torch.nn.MSELoss(reduction='none')
        loss = 0.0
                  
        
        for i in range(len(self.all_combos)):
            t_list = list()
            y_next = self.forward(x, str(self.all_combos[i][0]))
            y_preds = torch.zeros(batch_size, len(self.all_combos[i]), n_dim).float().to(self.device)
            y_preds[:, 0] = y_next
            
            t = int(self.all_combos[i][0]/min(self.step_sizes))
            t_list.append(t)
            for j in range(1, len(self.all_combos[i])):
                y_next = self.forward(y_next, str(self.all_combos[i][j]))
                y_preds[:, j] = y_next
                t += int(self.all_combos[i][j]/min(self.step_sizes))
                t_list.append(t)
                
            loss += criterion(y_preds, ys[:, t_list, :]).mean()

        return loss.mean()
        
        
    def train_net_single(self, dataset, max_epoch, batch_size, w=1.0, lr=1e-3, model_path=None, print_every=1000, size='4'):
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
#         self.check_data_info(dataset)

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
            batch_x = dataset.train_x[new_idxs[:batch_size], :]
            batch_ys = dataset.train_ys[new_idxs[:batch_size], :, :]
            # =============== calculate losses ================
            train_loss = self.calculate_loss_single(batch_x, batch_ys, w=w, size=size)
            val_loss = self.calculate_loss_single(dataset.val_x, dataset.val_ys, w=w, size=size)
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
            
            if epoch % print_every == 0:
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

    def calculate_loss_single(self, x, ys, w=1.0, size='1'):
        """
        :param x: x batch, array of size batch_size x n_dim
        :param ys: ys batch, array of size batch_size x n_steps x n_dim
        :return: overall loss
        """
        batch_size, n_steps, n_dim = ys.size()
        assert n_dim == self.n_dim

        # forward (recurrence)
        y_preds = torch.zeros(batch_size, n_steps, n_dim).float().to(self.device)
        y_prev = x
        for t in range(n_steps):
            y_next = self.forward(y_prev, size)
            y_preds[:, t, :] = y_next
            y_prev = y_next

        # compute loss
        criterion = torch.nn.MSELoss(reduction='none')
        loss = w * criterion(y_preds, ys).mean() + (1-w) * criterion(y_preds, ys).max()

        return loss


    def vectorized_multi_scale_forecast(self, x_init, n_steps, models, step_sizes=[8, 4]):
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
        extended_n_steps = n_steps + models[0].step_size
        preds = torch.zeros(n_test, extended_n_steps + 1, n_dim).float().to(device)

        # vectorized simulation
        indices.append(0)
        preds[:, 0, :] = x_init
        total_step_sizes = n_steps
#         for model in models:
        size_models = ['large', 'small']
        for i in [0, 1]:
            step_size = step_sizes[i]
            size_model = size_models[i]
            n_forward = int(total_step_sizes/step_size)
            y_prev = preds[:, indices, :].reshape(-1, n_dim)
            indices_lists = [indices]
            for t in range(n_forward):
                y_next = self.forward(y_prev, size_model)
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
            last_idx += step_size[-1]
            size_model = size_models[-1]
            y_next = self.forward(y_prev, size_model)
            preds[:, last_idx, :] = y_next
            indices.append(last_idx)
            y_prev = y_next

        # interpolations
        sample_steps = range(1, n_steps+1)
        valid_preds = preds[:, indices, :].detach().numpy()
        cs = scipy.interpolate.interp1d(indices, valid_preds, kind='lin/ear', axis=1)
        y_preds = torch.tensor(cs(sample_steps)).float()

        return y_preds

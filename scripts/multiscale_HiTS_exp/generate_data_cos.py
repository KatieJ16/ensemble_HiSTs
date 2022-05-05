import os
import numpy as np
import scipy as sp
from scipy import integrate
import matplotlib.pyplot as plt

# paths
data_dir = '../../data/'
hyperbolic_dir = os.path.join(data_dir, 'Hyperbolic')
cubic_dir = os.path.join(data_dir, 'Cubic')
vdp_dir = os.path.join(data_dir, 'VanDerPol')
hopf_dir = os.path.join(data_dir, 'Hopf')
lorenz_dir = os.path.join(data_dir, 'Lorenz')
lorenz_dir = os.path.join(data_dir, 'cos')

# adjustable parameters
dt = 0.01       # set to 5e-4 for Lorenz

#     noise = 0.05     # for study of noisy measurements, we use noise=0.01, 0.02; otherwise we leave it as 0.
n_forward = 5
total_steps = 1024 * n_forward
t = np.linspace(0, (total_steps)*dt, total_steps+1)

# system
#     noise = 0.2     # for study of noisy measurements, we use noise=0.01, 0.02; otherwise we leave it as 0.

mu = 2.0
def van_der_pol_rhs(x):
    return np.array([x[1], mu*(1-x[0]**2)*x[1]-x[0]])

# simulation parameters
np.random.seed(2)
n = 2

# dataset 
n_train = 3200
n_val = 320
n_test = 320


# simulate training trials 
train_data = np.zeros((n_train, total_steps+1, n))
print('generating training trials ...')
for i in range(n_train):
    x_init = [np.random.uniform(-2.0, 2.0), np.random.uniform(-4.0, 4.0)]
    sol = sp.integrate.solve_ivp(lambda _, x: van_der_pol_rhs(x), [0, total_steps*dt], x_init, t_eval=t)
    train_data[i, :, :] = sol.y.T

# simulate validation trials 
val_data = np.zeros((n_val, total_steps+1, n))
print('generating validation trials ...')
for i in range(n_val):
    x_init = [np.random.uniform(-2.0, 2.0), np.random.uniform(-2.0, 2.0)]    # make sure we have seen them in training set
    sol = sp.integrate.solve_ivp(lambda _, x: van_der_pol_rhs(x), [0, total_steps*dt], x_init, t_eval=t)
    val_data[i, :, :] = sol.y.T

# simulate test trials
test_data = np.zeros((n_test, total_steps+1, n))
print('generating testing trials ...')
for i in range(n_test):
    x_init = [np.random.uniform(-2.0, 2.0), np.random.uniform(-2.0, 2.0)]
    sol = sp.integrate.solve_ivp(lambda _, x: van_der_pol_rhs(x), [0, total_steps*dt], x_init, t_eval=t)
    test_data[i, :, :] = sol.y.T
    
for noise in [0,0.01,0.02, 0.05, 0.1, 0.2]:
    print("noise = ", noise)
    # add noise
    train_data_new = train_data + noise*train_data.std(1).mean(0)*np.random.randn(*train_data.shape)
    val_data_new = val_data + noise*val_data.std(1).mean(0)*np.random.randn(*val_data.shape)
    test_data_new = test_data + noise*test_data.std(1).mean(0)*np.random.randn(*test_data.shape)

    # save data
    np.save(os.path.join(vdp_dir, 'train_noise{}.npy'.format(noise)), train_data_new)
    np.save(os.path.join(vdp_dir, 'val_noise{}.npy'.format(noise)), val_data_new)
    np.save(os.path.join(vdp_dir, 'test_noise{}.npy'.format(noise)), test_data_new)
import numpy as np

x16 = np.load('all_combos_16.npy', allow_pickle=True)

x8 = []
x4 = []
x2 = []
for i in x16:
    this8 = []
    this4 = []
    this2 = []
    for j in i:
        this8.append(int(j/2))
        this4.append(int(j/4))
        this2.append(int(j/8))
    x8.append(this8)
    x4.append(this4)
    x2.append(this2)
    
np.save('all_combos_8.npy', x8)
np.save('all_combos_2.npy', x2)
np.save('all_combos_4.npy', x4)
import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize as imresize
from tqdm import tqdm
import torch

# In[]
with h5py.File('usps.h5', 'r') as hf:
        train = hf.get('train')
        X_tr = train.get('data')[:]
        y_tr = train.get('target')[:]
        test = hf.get('test')
        X_te = test.get('data')[:]
        y_te = test.get('target')[:]

# In[]
temp = np.reshape(X_te, (2007, 16, 16))  # Reshape and scale to appropriate range

# In[]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(temp[10, :, :])
ax.set_aspect('equal')
fig.show()

# In[]
new_temp = np.zeros((2007, 28, 28))
for n in tqdm(range(2007)):
    new_temp[n, :, :] = imresize(temp[n, :, :], (28, 28))

# In[]                  
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(new_temp[10, :, :])
ax.set_aspect('equal')
fig.show()

# In[]

# Convert to torch to make everybody happy
X = torch.from_numpy(new_temp)
Y = torch.from_numpy(y_te)

torch.save((X, Y), 'usps_test_data')
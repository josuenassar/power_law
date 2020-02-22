from ModelDefs.models import ModelFactory
from DataDefs.data import get_data
from tqdm import tqdm
import torch
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# In[]
"Create model"
# kwargs = {"dims": [(28 * 28, 1000), (1000, 1000), (1000, 1000), (1000, 10)], "activation": "relu", "architecture": "mlp",
#           "trainer": "vanilla", "regularizer": "eig", 'alpha_spectra': 1e-3, 'alpha_jacob':1e-4,
#           'optimizer': 'adam', 'lr': 1e-3, 'weight_decay': 0, 'cuda': False, 'eps': 0.3,
#           'gradSteps': 40, 'noRestarts': 1, 'lr_pgd': 1e-2,  'training_type': 'FGSM', 'slope': [1.04, 1.02, 1.0]}
kwargs = {"dims": [1, (28 * 28, 100), (100, 100), (100, 100), (100, 10)], "activation": "tanh", "architecture": "flat",
          "trainer": "vanilla", "regularizer": "no", 'alpha_spectra': 1e-3, 'alpha_jacob':1e-4,
          'optimizer': 'adam', 'lr': 1e-3, 'weight_decay': 0, 'cuda': False, 'eps': 0.3,
          'gradSteps': 40, 'noRestarts': 1, 'lr_pgd': 1e-2,  'training_type': 'FGSM', 'slope': [1.04, 1.02, 1.0]}
# kwargs = {"dims":[2, (1, 32), (32, 64), (1024, 1024), (1024, 10)], "activation": "tanh", "architecture": "mlp",
#           "trainer": "adv", "regularizer": "eigjac", 'alpha_spectra': 1e-3, 'alpha_jacob':1e-4,
#           'optimizer': 'adam', 'lr': 1e-3, 'weight_decay': 0, 'cuda': False, 'eps': 0.3,
#           'gradSteps': 40, 'noRestarts': 1, 'lr_pgd': 1e-2,  'training_type': 'PGD', 'only_last': True}
model = ModelFactory(**kwargs)

# In[]
"Get data loader"
train_loader, test_loader, full_loader = get_data(dataset='MNIST', batch_size=256, _seed=0,
                                                  validate=False, data_dir='')
# X_full, Y_full = next(iter(full_loader))
# In[]
for epoch in tqdm(range(100)):
    model.train_epoch(train_loader)
    # print(torch.cuda.memory_allocated())
    torch.cuda.empty_cache()
    # print(torch.cuda.memory_allocated())


# In[]
X_full, Y_full = next(iter(full_loader))

# In[]
model.eval()
hidden, _ = model.bothOutputs(X_full)

# In[]

from numpy import newaxis as na
temp = hidden[1].detach().numpy()
temp = temp - np.mean(temp, 0)
cov = temp.T @ temp / temp.shape[0]

# In[]
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.matshow(cov)
fig.show()

# In[]
print(1 - model.evaluate_dataset_test_loss(train_loader)[1])
print(1 - model.evaluate_dataset_test_loss(test_loader)[1])
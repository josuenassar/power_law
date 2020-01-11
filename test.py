from ModelDefs.models import ModelFactory
from DataDefs.data import get_data
from tqdm import tqdm
# In[]
"Create model"
kwargs = {"dims": [(28 * 28, 1000), (1000, 10)], "activation": "relu", "architecture": "mlp",
          "trainer": "vanilla", "regularizer": "eigjac", 'alpha_spectra': 1e-3, 'alpha_jacob':1e-4,
          'optimizer': 'adam', 'lr': 1e-3, 'weight_decay': 0, 'cuda': False, 'eps': 0.3,
          'gradSteps': 40, 'noRestarts': 1, 'lr_pgd': 1e-2,  'training_type': 'PGD'}
model = ModelFactory(**kwargs)

# In[]
"Get data loader"
train_loader, test_loader, full_loader = get_data(dataset='MNIST', batch_size=100, _seed=0,
                                                  validate=False, data_dir='')
X_full, Y_full = next(iter(full_loader))
# In[]
for epoch in tqdm(range(1)):
    model.train_epoch(train_loader, X_full)
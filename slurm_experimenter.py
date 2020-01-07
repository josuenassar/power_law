import os
from itertools import product, cycle, count
import random
from random_words import RandomWords
from reshape_data import download_and_convert
from socket import gethostname
import numpy as np
import collections
import json

depends = collections.defaultdict(dict)

env_name = "powwer_law"

rw = RandomWords()
homedir = os.path.expanduser("~")
job_directory = os.path.join(homedir, 'batch/')
os.makedirs(job_directory, exist_ok=True)
log_dir = os.path.join(homedir, 'logs/')
os.makedirs(log_dir, exist_ok=True)

if gethostname() == 'dirac':
    hostname = 'dirac'
elif gethostname() == 'login1':
    hostname = 'seawulf'
else:
    hostname = 'orion'

project_dir = os.path.join(homedir, 'projects/power_law_nn')
data_dir = os.path.join(project_dir, 'data')
result_dir = os.path.join(project_dir, 'result')
os.makedirs(result_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

architectures = [[(28 * 28, 200), (200, 200), (200, 10)],
                 [1, (1, 12), (1728, 1024), (1024, 10)]]
typesOfTraining = [False, True]
# activations = ['relu', 'tanh']
activations = ['relu']
# alphas = [0.1, 1, 5, 10]
alphas = 10**np.linspace(-3, 2, 10)
download_and_convert(data_dir)
data_dir = os.path.join(data_dir, 'mnist.npy')

queue_name = cycle(['gpu', 'gpu-large'])
queue_time = cycle(['8', '8'])
with open('actual_regularization/run_dependencies.json') as f:
    dependency_dict = json.load(f)
# for _, element in dependency_dict.items():
#     print(element['dims'])
#     if not isinstance(['dims'][0], int):
#         import pdb; pdb.set_trace()
#         element['dims'] = [tuple(k) if  not isinstance(k, int) else k for k in element['dims']]
#     else:
#         element['dims'] = [tuple(k) for k in element['dims']]
for arch, act, alpha in product(architectures, activations, alphas):
    randname = rw.random_word() + str(random.randint(0, 100))
    job_file = os.path.join(job_directory, "{}.job".format(randname))
    if isinstance(arch[0], int):
        model = 'cnn'
        batch_size = int(arch[2][0] * 1.5)
    else:
        model = 'mlp'
        batch_size = int(arch[1][0] * 1.5)
    with open(job_file, 'w+') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --job-name={}.job\n".format(randname))
        fh.writelines("#SBATCH --output={}/{}.out\n".format(log_dir, randname))
        fh.writelines("#SBATCH --error={}/{}.err\n".format(log_dir, randname))
        fh.writelines("#SBATCH --ntasks-per-node=8\n")
        # fh.writelines("#SBATCH --cpus-per-task=24\n")
        fh.writelines("#SBATCH --nodes=1\n")
        fh.writelines("#SBATCH --time={}:00:00\n".format(next(queue_time)))
        fh.writelines("#SBATCH --partition={}\n".format(next(queue_name)))
        fh.writelines("module load shared\n")
        fh.writelines("module load anaconda/3\n")
        fh.writelines("module load cuda100/toolkit/10.0\n")
        fh.writelines("module load cudnn/7.0.5\n")
        fh.writelines("source /gpfs/software/Anaconda3/bin/activate {}\n".format(env_name))
        fh.writelines("cd \n")
        fh.writelines("cd {}/actual_regularization \n".format(project_dir))
        fh.writelines("#SBATCH --mail-type=ALL\n")
        fh.writelines("#SBATCH --mail-user=$piotr.sokol@stonybrook.edu\n")
        fh.writelines("python ray_experimenter.py --dims \'{}\' --host {} --pathLoad {} --pathSave {} "
                      "--activation {} --alpha {} --batch_size {} --modelType {}\n".format(arch,
                                                                                           hostname,
                                                                                           data_dir, result_dir, act, alpha,
                                                                                           batch_size, model))
    dependency_name = None
    for _, element in dependency_dict.items():
        element['dims'] = [tuple(k) if  not isinstance(k, int) else k for k in element['dims']]
        if element['modelType'] == model and element['dims'] == arch and element['batch_size'] == batch_size and element['activation'] == act and element['alpha'] == alpha:
            dependency_name = element['jobName']
            break
    assert dependency_name is not None

    # jobid = os.popen("sacct -n -X --format jobid --name {}.job".format(dependency_name)).read().strip().strip('\n')
    # os.system("sbatch --dependency=afterok:{} {}".format(jobid, job_file))

for arch,act in product(architectures, activations):
    eps = 0.3
    randname = rw.random_word() + str(random.randint(0, 100))
    job_file = os.path.join(job_directory, "{}.job".format(randname))
    if isinstance(arch[0], int):
        model = 'cnn'
        batch_size = int(arch[2][0] * 1.5)
    else:
        model = 'mlp'
        batch_size = int(arch[1][0] * 1.5)
    with open(job_file, 'w+') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --job-name={}.job\n".format(randname))
        fh.writelines("#SBATCH --output={}/{}.out\n".format(log_dir, randname))
        fh.writelines("#SBATCH --error={}/{}.err\n".format(log_dir, randname))
        fh.writelines("#SBATCH --ntasks-per-node=8\n")
        # fh.writelines("#SBATCH --cpus-per-task=24\n")
        fh.writelines("#SBATCH --nodes=1\n")
        fh.writelines("#SBATCH --time={}:00:00\n".format(next(queue_time)))
        fh.writelines("#SBATCH --partition={}\n".format(next(queue_name)))
        fh.writelines("module load shared\n")
        fh.writelines("module load anaconda/3\n")
        fh.writelines("module load cuda100/toolkit/10.0\n")
        fh.writelines("module load cudnn/7.0.5\n")
        fh.writelines("source /gpfs/software/Anaconda3/bin/activate {}\n".format(env_name))
        fh.writelines("cd \n")
        fh.writelines("cd {}/actual_regularization \n".format(project_dir))
        fh.writelines("#SBATCH --mail-type=ALL\n")
        fh.writelines("#SBATCH --mail-user=$piotr.sokol@stonybrook.edu\n")

        fh.writelines("python ray_experimenter.py --dims \'{}\' --host {} --pathLoad {} --pathSave {} --activation {} "
                      "--eps {} --alpha {} --batch_size {} --modelType {}\n".format(
            arch, hostname, data_dir, result_dir, act, eps, 0.01, batch_size, model))
    dependency_name = None
    for _, element in dependency_dict.items():
        # print(element['dims'])
        element['dims'] = [tuple(k) if not isinstance(k, int) else k for k in element['dims']]
        if element['modelType'] == model and element['dims'] == arch and element['batch_size'] == batch_size and \
                element['activation'] == act and element['alpha'] == alpha:
            dependency_name = element['jobName']
    assert dependency_name is not None
    jobid = os.popen("sacct -n -X --format jobid --name {}.job".format(dependency_name)).read().strip().strip('\n')
    os.system("sbatch --dependency=afterok:{} {}".format(jobid, job_file))

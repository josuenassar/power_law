# Requires pymongo 3.6.0+
from bson.tz_util import FixedOffset
from datetime import datetime
from pymongo import MongoClient
import os
from itertools import product
from redis import Redis
from rq import Queue, Worker, Connection
import subprocess
from worker import gen_arg_list, subprocs_exec
import uuid
import socket
import sys, os
sys.path.insert(0, os.path.abspath('..'))
BASEPATH = os.path.dirname(os.getcwd())
data_dir = os.path.join(BASEPATH, 'data')  # TODO fill this and set it as an abspath

# hostname = 'erdos'
# regularization = ["eigjac", "eig"]
# trainer = ["adv", "vanilla"]
# alpha_spectra = [1e-4, 1e-3, 1e-2, 1e-1]
# stuff_to_loop_over = list(product(regularization, trainer, alpha_spectra))
# Make top level directories
if socket.gethostname() in ['dirac']:
    hostname = 'dirac'
    regularization = ["jac"]
    # regularization = ["no"]
    trainer = ["adv", "vanilla"]
    alpha_jacob = 0.01
    stuff_to_loop_over = list(product(regularization, trainer))
elif socket.gethostname() == 'erdos':
    hostname = 'erdos'
    regularization = ["eig"]
    trainer = ["vanilla", 'adv']
    last_layer = [True, False]
    alpha_spectra = [1e-3, 1e-2, 1e-1, 1, 2, 5]
    stuff_to_loop_over = list(product(regularization, trainer, alpha_spectra, last_layer))
elif socket.gethostname() == 'catniplab-Alienware':
    hostname = 'catniplab-Alienware'
    regularization = ["no"]
    trainer = ["adv", "vanilla"]
    stuff_to_loop_over = list(product(regularization, trainer))
else:
    hostname = 'catniplab-Alienware'
    regularization = ["no"]
    trainer = ["adv", "vanilla"]
    stuff_to_loop_over = list(product(regularization, trainer))

# Created with Studio 3T, the IDE for MongoDB - https://studio3t.com/
todos = []  # list of config files :D
reps = 1  # number of realizations of each architecture to train

# In[]
connection = Redis()
gpu = Queue('gpu', connection=connection)
gpu0 = Queue('gpu0', connection=connection)
gpu1 = Queue('gpu1', connection=connection)

queue = gpu
# tmp_dir = '/tmp/rq_{}/'.format(uid)
lr = 1e-4

for stuff in stuff_to_loop_over:
    uid = str(uuid.uuid4())[:8]
    tmp_dir = '/tmp/rq_{}/'.format(uid)
    arg = gen_arg_list(["rsync", BASEPATH, tmp_dir[:-1], "-r", "-l"], {"exclude": '"MNIST/ data/ '
                                                                                  'empirical_experiments/ '
                                                                                  'analyze_networks/ USPS/"'})
    subprocess.call(" ".join(arg), shell=True)

    if socket.gethostname() in ['dirac', 'catniplab-Alienware']:
        reg, tr = stuff
        cmds = ['python',
                tmp_dir + 'power_law/ray_training_experimenter.py',
                '--architecture', 'FC',
                '--save_dir', str(BASEPATH),
                '--data_dir', str(data_dir),
                '--regularizer', reg,
                '--trainer', tr,
                '--alpha_jacob', str(alpha_jacob),
                '--reps', str(reps),
                ';', 'rm', '-rf', tmp_dir]
        # cmds = ['python','-c','import time;','time.sleep(30)']
    elif socket.gethostname() == 'erdos':
        reg, tr, alpha, layer = stuff
        cmds = ['python',
                tmp_dir + 'power_law/ray_training_experimenter.py',
                '--architecture', 'FC',
                '--save_dir', str(BASEPATH),
                '--data_dir', str(data_dir),
                '--regularizer', reg,
                '--trainer', tr,
                '--alpha_spectra', str(alpha),
                '--only_last', str(layer),
                '--reps', str(reps),
                ';', 'rm', '-rf', tmp_dir]

    kwargs = {"dir": os.path.join(tmp_dir, 'power_law')}

    ttl = None  # infinite time in queue
    queue.enqueue_call(func=subprocs_exec,
                        args=(cmds,kwargs),
                        timeout='14d', ttl=ttl,
                        result_ttl=ttl,
                       at_front=False)

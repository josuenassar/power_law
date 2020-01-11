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

# Make top level directories
if socket.gethostname() in ['dirac']:
    regularization = ["jac"]
    trainer = ["adv", "vanilla"]
    stuff_to_loop_over = product(regularization, trainer)
elif socket.gethostname() == 'erdos':
    regularization = ["eigjac", "eig"]
    trainer = ["adv", "vanilla"]
    alpha_spectra = [1e-4, 1e-3, 1e-2, 1e-1]
    stuff_to_loop_over = product(regularization, trainer, alpha_spectra)
elif socket.gethostname() == 'catniplab-Alienware':
    regularization = ["no"]
    trainer = ["adv", "vanilla"]
    stuff_to_loop_over = product(regularization, trainer)

connection=Redis()
gpu = Queue('gpu',connection=connection)
gpu0 = Queue('gpu0',connection=connection)
gpu1 = Queue('gpu1',connection=connection)

queue = gpu
# tmp_dir = '/tmp/rq_{}/'.format(uid)


for stuff in stuff_to_loop_over:
    uid = str(uuid.uuid4())[:8]
    tmp_dir = '/tmp/rq_{}/'.format(uid)
    arg = gen_arg_list(["rsync", BASEPATH, tmp_dir[:-1], "-r", "-l"], {"exclude": '"data/"'})
    subprocess.call(" ".join(arg), shell=True)

    if socket.gethostname() in ['dirac', 'catniplab-Alienware']:
        reg, tr = stuff
        cmds = ['python',
                tmp_dir + 'power_law/ray_hyperparam_experimenter.py',
                '--architecture', 'MadryMNIST',
                '--save_dir', str(BASEPATH),
                '--data_dir', str(data_dir),
                '--regularizer', str(reg), '--trainer', str(tr),
                ';', 'rm', '-rf', tmp_dir]
        # cmds = ['python','-c','import time;','time.sleep(30)']
    elif socket.gethostname() == 'erdos':
        reg, tr, alpha = stuff
        cmds = ['python',
                tmp_dir + 'power_law/ray_hyperparam_experimenter.py',
                '--architecture', 'MadryMNIST',
                '--save_dir', str(BASEPATH),
                '--data_dir', str(data_dir),
                '--regularizer', str(reg), '--trainer', str(tr), '--alpha_spectra', str(alpha),
                ';', 'rm', '-rf', tmp_dir]

    kwargs = { "dir": os.path.join(tmp_dir,'power_law')}

    ttl = None  # infinite time in queue
    queue.enqueue_call(func=subprocs_exec,
                        args=(cmds,kwargs),
                        timeout='14d', ttl=ttl,
                        result_ttl=ttl,
                       at_front=False)

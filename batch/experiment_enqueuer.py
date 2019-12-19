import os
from itertools import product
import random
from random_words import RandomWords
rw = RandomWords()
currdir=os.getcwd()
job_directory = os.path.join(currdir, 'batch/')
# import pdb; pdb.set_trace()
# Make top level directories

datasets = ['CIFAR10']
manifolds = ['stiefel','oblique','euclidean']
h0s = [8, 0.5]

import sys, os
sys.path.insert(0, os.path.abspath('..'))

from redis import Redis
from rq import Queue, Worker, Connection
import subprocess
from worker import gen_arg_list, subprocs_exec

connection=Redis()

gpu = Queue('gpu',connection=connection)
gpu0 = Queue('gpu0',connection=connection)
gpu1 = Queue('gpu1',connection=connection)

queue = gpu

import uuid
uid = str(uuid.uuid4())[:8]
tmp_dir = '/tmp/rq_{}/'.format(uid)
BASEPATH = '/home/piotr/projects/isonetry'
arg = gen_arg_list(["rsync", BASEPATH, tmp_dir[:-1], "-r", "-l"], {"exclude": '"data/"'})
subprocess.call(" ".join(arg), shell=True)

cmds =[]
kwargs = { "dir": os.path.join(tmp_dir,'isonetry')}

for rep in range(5):
    for dataset,manifold, h0 in product(datasets, manifolds, h0s):
        uid = str(uuid.uuid4())[:8]
        tmp_dir = '/tmp/rq_{}/'.format(uid)
        BASEPATH = '/home/piotr/projects/isonetry'
        arg = gen_arg_list(["rsync", BASEPATH, tmp_dir[:-1], "-r", "-l"], {"exclude": '"data/"'})
        subprocess.call(" ".join(arg), shell=True)

        cmds =[]
        kwargs = { "dir": os.path.join(tmp_dir,'isonetry')}


        cmds = ['python', tmp_dir + 'isonetry/experiment/experiment_runner.py', dataset,manifold, str(h0), ';', 'rm',
                '-rf', tmp_dir]
        ttl = 1# infinite time in queue
        queue.enqueue_call(func=subprocs_exec,
                            args=(cmds,kwargs),
                            timeout='7d', ttl=ttl,
                            result_ttl=ttl,
                           at_front=False)

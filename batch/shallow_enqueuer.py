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
import fire
sys.path.insert(0, os.path.abspath('..'))
BASEPATH = os.path.dirname(os.getcwd())
data_dir = os.path.join(BASEPATH, 'data')  # TODO fill this and set it as an abspath


def run(alpha_spectra=0):
    trainer = ['vanilla']
    if alpha_spectra == 0:
        regularization = ['no']
    else:
        regularization = ['eig']
    stuff_to_loop_over = list(product(regularization, trainer))
    reps = 2
    lr = 1e-4
    hostname = socket.gethostname()

    # In[]
    connection = Redis()
    gpu = Queue('gpu', connection=connection)
    gpu0 = Queue('gpu0', connection=connection)
    gpu1 = Queue('gpu1', connection=connection)

    queue = gpu

    for stuff in stuff_to_loop_over:
        uid = str(uuid.uuid4())[:8]
        if hostname == 'erdos':
            tmp_dir = '/scratch/rq_{}/'.format(uid)
        else:
            tmp_dir = '/tmp/rq_{}/'.format(uid)
        arg = gen_arg_list(["rsync", BASEPATH, tmp_dir[:-1], "-av", "-l", '--exclude={MNIST,data,emperical_experiments,'
                                                                                                       'analyze_networks,USPS}'])
        #import pdb; pdb.set_trace()
        subprocess.call(" ".join(arg), shell=True)
        reg, tr = stuff
        cmds = ['python',
                tmp_dir + 'power_law/ray_training_experimenter.py',
                '--architecture', 'Shallow',
                '--save_dir', str(BASEPATH),
                '--data_dir', str(data_dir),
                '--regularizer', reg,
                '--trainer', tr,
                '--alpha_spectra', str(alpha_spectra),
                '--reps', str(reps),
                ';', 'rm', '-rf', tmp_dir]

        kwargs = {"dir": os.path.join(tmp_dir, 'power_law')}

        ttl = None  # infinite time in queue
        queue.enqueue_call(func=subprocs_exec,
                            args=(cmds,kwargs),
                            timeout='14d', ttl=ttl,
                            result_ttl=ttl,
                           at_front=False)


if __name__ == '__main__':
  fire.Fire(run)

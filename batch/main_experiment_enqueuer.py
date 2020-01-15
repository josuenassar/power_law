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
    trainer = ["adv", "vanilla"]
    stuff_to_loop_over = list(product(regularization, trainer))
elif socket.gethostname() == 'erdos':
    hostname = 'erdos'
    regularization = ["eigjac", "eig"]
    trainer = ["adv", "vanilla"]
    alpha_spectra = [1e-4, 1e-3, 1e-2, 1e-1]
    stuff_to_loop_over = list(product(regularization, trainer, alpha_spectra))
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

client = MongoClient('mongodb://powerLawNN:Pareto_a^-b@ackermann.memming.com/admin?authMechanism=SCRAM-SHA-1')
database = client["powerLawHypers"]
collection = database["runs"]
# Created with Studio 3T, the IDE for MongoDB - https://studio3t.com/
todos = []  # list of config files :D
reps = 3  # number of realizations of each architecture to train

# In[]
"Construct pipeline for qu"
pipeline = [
    {
        u"$match": {
            u"start_time": {
                u"$gte": datetime.strptime("2020-01-10 19:00:00.000000", "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo = FixedOffset(-300, "-0500"))
            },
            u"status": u"COMPLETED"
        }
    },
    {
        u"$group": {
            u"_id": {
                u"server": u"$host.hostname",
                u"trainer": u"$config.trainer",
                u"regularizer": u"$config.regularizer",
                u"alpha_spectra": u"$config.alpha_spectra",
            },
            u"maxAcc": {
                u"$max": u"$result"
            },
            u"count": {
                u"$sum": 1.0
            }
        }
    },
    {
        u"$match": {
            u"count": {
                u"$gte": 50.0
            }
        }
    }
]
cursor = collection.aggregate(
    pipeline,
    allowDiskUse=False
)
count = 0
try:
    for doc in cursor:
        for stuff in stuff_to_loop_over:
            if hostname == 'erdos':
                reg, tr, alpha = stuff
                if doc["_id"]["trainer"] == tr and doc["_id"]["server"] == hostname and doc["_id"][
                    "regularizer"] == reg and doc["_id"]["alpha_spectra"] == alpha:
                    count += 1
                    query = {}
                    query["result"] = doc['maxAcc']
                    print(doc['maxAcc'])
                    query["config.alpha_spectra"] = doc["_id"]["alpha_spectra"]
                    query["config.trainer"] = doc["_id"]["trainer"]
                    query["config.regularizer"] = doc["_id"]["regularizer"]
                    cfg = collection.find(query).limit(1)[0]['config']
                    todos.append(cfg)
            else:
                reg, tr = stuff
                if doc["_id"]["trainer"] == tr and doc["_id"]["server"] == hostname and doc["_id"][
                    "regularizer"] == reg:
                    count += 1
                    query = {}
                    query["result"] = doc['maxAcc']
                    print(doc['maxAcc'])
                    query["config.alpha_spectra"] = doc["_id"]["alpha_spectra"]
                    query["config.trainer"] = doc["_id"]["trainer"]
                    query["config.regularizer"] = doc["_id"]["regularizer"]
                    cfg = collection.find(query).limit(1)[0]['config']
                    todos.append(cfg)


finally:
    client.close()

connection = Redis()
gpu = Queue('gpu', connection=connection)
gpu0 = Queue('gpu0', connection=connection)
gpu1 = Queue('gpu1', connection=connection)

queue = gpu
# tmp_dir = '/tmp/rq_{}/'.format(uid)


# for stuff in stuff_to_loop_over:
for task in todos:
    uid = str(uuid.uuid4())[:8]
    tmp_dir = '/tmp/rq_{}/'.format(uid)
    arg = gen_arg_list(["rsync", BASEPATH, tmp_dir[:-1], "-r", "-l"], {"exclude": '"data/"'})
    subprocess.call(" ".join(arg), shell=True)

    if socket.gethostname() in ['dirac', 'catniplab-Alienware']:
        cmds = ['python',
                tmp_dir + 'power_law/ray_training_experimenter.py',
                '--architecture', 'MadryMNIST',
                '--save_dir', str(BASEPATH),
                '--data_dir', str(data_dir),
                '--regularizer', str(task['regularizer']),
                '--trainer', str(task['trainer']),
                '--lr', str(task['lr']),
                ';', 'rm', '-rf', tmp_dir]
        # cmds = ['python','-c','import time;','time.sleep(30)']
    elif socket.gethostname() == 'erdos':
        reg, tr, alpha = stuff
        cmds = ['python',
                tmp_dir + 'power_law/ray_training_experimenter.py',
                '--architecture', 'MadryMNIST',
                '--save_dir', str(BASEPATH),
                '--data_dir', str(data_dir),
                '--regularizer', str(task['regularizer']),
                '--trainer', str(task['trainer']),
                '--alpha_spectra', str(task['alpha_spectra']),
                '--lr', str(task['lr']),
                ';', 'rm', '-rf', tmp_dir]

    kwargs = {"dir": os.path.join(tmp_dir, 'power_law')}

    ttl = None  # infinite time in queue
    queue.enqueue_call(func=subprocs_exec,
                        args=(cmds,kwargs),
                        timeout='14d', ttl=ttl,
                        result_ttl=ttl,
                       at_front=False)

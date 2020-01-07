import ray
import ray.tune as tune
from ray.tune.schedulers import FIFOScheduler
from ray.tune.suggest.skopt import SkOptSearch
from skopt.space import Real
from skopt import Optimizer
from multiprocessing import cpu_count
from torch.cuda import device_count
import argparse
import uuid
import os
from sacred.observers import MongoObserver
from pymongo import MongoClient, DESCENDING



nCPU = cpu_count()
nGPU = device_count()
load = 1/2  # how large a fraction of the GPU memory does the model take up 0 <= load <=  1

argparser = argparse.ArgumentParser()
argparser.add_argument('--lr', type=float, default=argparse.SUPPRESS)
argparser.add_argument('--modelType',
                       choices=['mlp', 'cnn'],
                       default=argparse.SUPPRESS,
                       help='Type of neural network.')
argparser.add_argument('--activation',
                       choices=['relu', 'tanh'],
                       help='Nonlinearity.',default=argparse.SUPPRESS)
argparser.add_argument('--advTraining', action='store_true')
argparser.add_argument('--pathSave', type=str, required=True)
argparser.add_argument('--pathLoad', type=str, required=True)
argparser.add_argument('--batch_size', type=int, default=argparse.SUPPRESS)
argparser.add_argument('--alpha', type=float, default=argparse.SUPPRESS)
argparser.add_argument('--gradSteps', type=int, default=argparse.SUPPRESS)
argparser.add_argument('--noRestarts', type=int, default=argparse.SUPPRESS)
argparser.add_argument('--numEpochs', default=100_000, type=int)
argparser.add_argument('--eps', type=float, default=argparse.SUPPRESS)
argparser.add_argument('--dims', type=str, default=argparse.SUPPRESS)
argparser.add_argument('--smoke-test', action="store_true", default=False)
argparser.add_argument('--host', choices=['seawulf', 'orion', 'dirac'], help='Server where the code ran',
                       default='seawulf')
argparser.add_argument('--reps', help='Server where the code ran', type=int,
                       default=5)
args, unknownargs = argparser.parse_known_args()

args = vars(args)
try :
    args['dims'] = eval(args['dims'])
except:
    pass
TT = args['advTraining']
smoke_test = args['smoke_test']
host = args['host']
[args.pop(k) for k in ['advTraining','smoke_test','host']]
args['pathSave'] = os.path.join(args['pathSave'], 'tmp' + str(uuid.uuid4())[:8])

# #TODO query mongoDB and create config
# client = MongoClient("mongodb://powerLawNN:Pareto_a^-b@ackermann.memming.com/admin?authMechanism=SCRAM-SHA-1")
# db = client.powerLawHypers

# if TT:
#     config = db['runs'].find({
#         "$and": [{"config.eps": args['eps']},
#                  {"config.alpha": args['alpha']},
#                  {"config.batch_size": args['batch_size']},
#                  {"config.modelType": args['modelType']}
#                  ]},
#         {
#             "config.lr": 1,
#         }
#     ).sort("result", DESCENDING).limit(1)[0]['config']
# else:
#     config = db['runs'].find({
#         "$and": [{"config.alpha": args['alpha']},
#                  {"config.batch_size": args['batch_size']},
#                  {"config.modelType": args['modelType']}
#                  ]},
#         {
#             "config.lr": 1,
#         }
#     ).sort("result", DESCENDING).limit(1)[0]['config']


@ray.remote(num_gpus=load, num_cpus=int(load * nCPU/nGPU))
def train():
    import time, random
    time.sleep(random.uniform(0.0, 10.0))
    if TT:
        from advTraining import ex
    else:
        from training import ex
    if smoke_test:
        config = {'numEpochs': 1,**config}
    mongoDBobserver = MongoObserver.create(
        url='mongodb://powerLawNN:Pareto_a^-b@ackermann.memming.com/admin?authMechanism=SCRAM-SHA-1',
        db_name='powerLawExpts')
    ex.observers.append(mongoDBobserver)
    ex.run(config_updates={**args, **config})
    result = ex.current_run.result

jobid  = os.popen("sacct -n -X --format jobid --name twists77.job").read().strip().strip('\n')
if __name__ == '__main__':
    import tweepy
    consumer_key = "emYE7xtZ3gdw48OuGyQ50xTXB"
    consumer_secret = "lEQRqnXLkT3Hh2V8fZ5gCOoUslJeXj0wzWnTupkabEgNkIVCOg"
    access_token = "1007403120899579904-veqrICIYFfSnfYtvWcb7oryiuWXc2h"
    access_token_secret = "eimBG9y1nd6bawHZGmwQf2dkQRhRPWK4R1X6yQ6eq0lZq"
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    # DO TRAINING & SAVE TO DB
    ray.init(num_cpus=nCPU//2, num_gpus=1)
    [train.remote() for i in range(args['reps'])]

    if TT:
        api.update_status(status="Evaluated {} adversarially trained models for a power_law_nn {} with {} layer dims "
                                 "on {}.".format(args['reps'], args['modelType'], str(args['dims']),host))
    else:
        api.update_status(status="Evaluated {} penalized models for a power_law_nn {} with {} layer dims "
                                 "on {}.".format(args['reps'], args['modelType'], str(args['dims']), host))
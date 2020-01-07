import ray
import ray.tune as tune
from ray.tune.schedulers import FIFOScheduler
from ray.tune.suggest.skopt import SkOptSearch
from skopt.space import Real
from skopt import Optimizer
# from multiprocessing import cpu_count
from torch.cuda import device_count
import argparse
# import uuid
# import os
import socket
from sacred.observers import MongoObserver


nGPU = device_count()

argparser = argparse.ArgumentParser()
argparser.add_argument("--activation",  choices=['relu', 'tanh'],
                       help='Nonlinearity.',default=argparse.SUPPRESS)
argparser.add_argument("--alpha_jacob",  type=float, default=argparse.SUPPRESS)
argparser.add_argument("--alpha_spectra",  type=float, default=argparse.SUPPRESS)
argparser.add_argument("--architecture", choices= ["LeNet5", "MadryMNIST", "mlp", "cnn"], required=True,
                       help='Type of neural network.')
argparser.add_argument("--cuda", default=True)
argparser.add_argument("--data_dir", required=True)
argparser.add_argument("--dataset", choices=["MNIST","CIFAR10"], default=argparse.SUPPRESS)
argparser.add_argument("--dims", type=str, default=argparse.SUPPRESS)
argparser.add_argument('--eps', type=float, default=argparse.SUPPRESS)
argparser.add_argument('--gradSteps', type=int, default=argparse.SUPPRESS)
argparser.add_argument("--lr", default=argparse.SUPPRESS)
argparser.add_argument("--lr_pgd", default=argparse.SUPPRESS)
argparser.add_argument("--max_epochs", type=int, default=argparse.SUPPRESS)
argparser.add_argument("--max_iter", type=int, default=argparse.SUPPRESS)
argparser.add_argument('--noRestarts', type=int, default=argparse.SUPPRESS)
argparser.add_argument("--optimizer", choices=["adam", "rms", "sgd"], default=argparse.SUPPRESS)
argparser.add_argument("--regularizer", choices=["no", "jac", "eig", "eigjac"], default="no")
argparser.add_argument("--save_dir", required=True)
argparser.add_argument("--trainer", choices= ["vanilla", "adv"], default="vanilla")
argparser.add_argument("--training_type", choices=["FGSM", "PGD"], default=argparse.SUPPRESS)
argparser.add_argument("--hpsearch", type=bool, default=True)
argparser.add_argument("--weight_decay", type=float, default=0.)
argparser.add_argument('--optimize', choices=['all','not_spectra'], default='all')

argparser.add_argument('--smoke-test', action="store_true", default=False)

args, unknownargs = argparser.parse_known_args()
args = vars(args)
try :
    args['dims'] = eval(args['dims'])
except:
    pass

smoke_test = args['smoke_test']
if args["architecture"] in ["LeNet5", "MadryMNIST"]:
    architecture = args["architecture"]
    what_to_optimize = args["optimize"]
    [args.pop(k) for k in ['architecture', 'optimize']]
    try:
        args.pop("dims")
    except:
        pass
args.pop('smoke_test')

def train(config, reporter):
    import time, random; time.sleep(random.uniform(0., 10.))
    from experiment import ex  # importing experiment here is crucial!
    if smoke_test:
        config = {'max_epochs': 1, **config}
    ex.observers.append(MongoObserver(
        url='mongodb://powerLawNN:Pareto_a^-b@ackermann.memming.com/admin?authMechanism=SCRAM-SHA-1',
        db_name='powerLawHypers'))
    ex.run(named_configs=[architecture], config_updates={**args, **config})
    result = ex.current_run.result
    reporter(result=result, done=True)

if __name__ == '__main__':
    import tweepy
    consumer_key = "emYE7xtZ3gdw48OuGyQ50xTXB"
    consumer_secret = "lEQRqnXLkT3Hh2V8fZ5gCOoUslJeXj0wzWnTupkabEgNkIVCOg"
    access_token = "1007403120899579904-veqrICIYFfSnfYtvWcb7oryiuWXc2h"
    access_token_secret = "eimBG9y1nd6bawHZGmwQf2dkQRhRPWK4R1X6yQ6eq0lZq"
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    budget = 1

    parameter_names = ['lr']
    space = [Real(10 ** -9, 10 ** -3, "log-uniform", name='lr')]
    if args['trainer'] == 'adv':
        pass
    if 'jac' in args['regularizer']:
        parameter_names.append('alpha_jacob')
        space.append([Real(10 ** -8, 10 ** 1, "log-uniform", name='alpha_jacob')])
    if 'eig' in args['regularizer'] and what_to_optimize == 'all':
        parameter_names.append('alpha_spectra')
        space.append([Real(10 ** -8, 10 ** 1, "log-uniform", name='alpha_spectra')])

    ray.init(num_gpus=nGPU)
    optimizer = Optimizer(
        dimensions=space,
        random_state=1,
        base_estimator='gp'
    )
    algo = SkOptSearch(
        optimizer,
        parameter_names=parameter_names,
        max_concurrent=4,
        metric="result",
        mode="max")

    scheduler = FIFOScheduler()
    tune.register_trainable("train_func", train)

    tune.run_experiments({
        'my_experiment': {
            'run': 'train_func',
            'resources_per_trial': {"gpu": 1},
            'num_samples': budget,
        }
    }, search_alg=algo, scheduler=scheduler)

    api.update_status(status="Finished hyperparam sweep for power_law on {}".format(socket.gethostname()))
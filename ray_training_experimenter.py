import ray
from multiprocessing import cpu_count
from torch.cuda import device_count
import argparse
import uuid
import os
from sacred.observers import MongoObserver



nCPU = cpu_count()
nGPU = device_count()
# load = 1/2  # how large a fraction of the GPU memory does the model take up 0 <= load <=  1

argparser = argparse.ArgumentParser()
argparser.add_argument("--activation",  choices=['relu', 'tanh'],
                       help='Nonlinearity.',default=argparse.SUPPRESS)
argparser.add_argument("--alpha_jacob",  type=float, default=argparse.SUPPRESS)
argparser.add_argument("--alpha_spectra",  type=float, default=argparse.SUPPRESS)
argparser.add_argument("--architecture", choices=["LeNet5", "MadryMNIST", "FC", "mlp", "cnn"], required=True,
                       help='Type of neural network.')
argparser.add_argument("--cuda", default=True)
argparser.add_argument("--data_dir", required=True)
argparser.add_argument("--dataset", choices=["MNIST", "CIFAR10"], default=argparse.SUPPRESS)
argparser.add_argument("--dims", type=str, default=argparse.SUPPRESS)
argparser.add_argument('--eps', type=float, default=argparse.SUPPRESS)
argparser.add_argument('--gradSteps', type=int, default=argparse.SUPPRESS)
argparser.add_argument("--lr", type=float, default=argparse.SUPPRESS)
argparser.add_argument("--lr_pgd", default=argparse.SUPPRESS)
argparser.add_argument("--max_epochs", type=int, default=500)
argparser.add_argument("--max_iter", type=int, default=argparse.SUPPRESS)
argparser.add_argument('--noRestarts', type=int, default=argparse.SUPPRESS)
argparser.add_argument("--optimizer", choices=["adam", "rms", "sgd"], default=argparse.SUPPRESS)
argparser.add_argument("--regularizer", choices=["no", "jac", "eig", "eigjac"], default="no")
argparser.add_argument("--save_dir", required=True)
argparser.add_argument("--trainer", choices= ["vanilla", "adv"], default="vanilla")
argparser.add_argument("--training_type", choices=["FGSM", "PGD"], default=argparse.SUPPRESS)
argparser.add_argument("--hpsearch", type=bool, default=False)
argparser.add_argument("--weight_decay", type=float, default=0.)
argparser.add_argument("--only_last", type=bool, default=False)
argparser.add_argument('--reps', help='Server where the code ran', type=int,
                       default=3)

argparser.add_argument('--smoke-test', action="store_true", default=False)
args, unknownargs = argparser.parse_known_args()
args = vars(args)
try :
    args['dims'] = eval(args['dims'])
except:
    pass

smoke_test = args['smoke_test']
if args["architecture"] in ["LeNet5", "MadryMNIST", "FC"]:
    architecture = args["architecture"]
    [args.pop(k) for k in ['architecture']]
    try:
        args.pop("dims")
    except:
        pass
args.pop('smoke_test')
reps = args['reps']
args.pop('reps')

@ray.remote(num_gpus=1)
def train():
    import time, random
    import importlib
    import experiment
    importlib.reload(experiment)
    from experiment import ex  # importing experiment here is crucial!

    time.sleep(random.uniform(0.0, 10.0))
    if smoke_test:
        config = {'max_epochs': 1}
    else:
        config = {}
    mongoDBobserver = MongoObserver(
        url='mongodb://powerLawNN:Pareto_a^-b@ackermann.memming.com/admin?authMechanism=SCRAM-SHA-1',
        db_name='powerLawExpts')
    ex.observers.append(mongoDBobserver)
    ex.run(named_configs=[architecture], config_updates={**args, **config})
    result = ex.current_run.result


if __name__ == '__main__':
    ray.init(num_gpus=nGPU)
    futures =[train.remote() for i in range(reps)]
    print(ray.get(futures))

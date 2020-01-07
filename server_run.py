from joblib import Parallel, delayed
import sys
sys.path.append('../ModelDefs/')
import training_script
import fire


def money(cuda=False):
    seeds = [i for i in range(3)]
    eigs = [0, 1e-1, 1e-2, 1e-3, 1e-4]
    jacobs = [0, 1e-1, 1e-2, 1e-3, 1e-4]
    eps = [0, abs((0.25 - 1307) / 3081)]
    activations = ['tanh', 'relu']
    dims = [2, (1, 32), (32, 64), (1024, 1024), (1024, 10)]
    # In[]
    "Run code"
    for eig in eigs:
        for jacob in jacobs:
            for ep in eps:
                for nonlin in activations:
                    Parallel(n_jobs=2)(delayed(training_script.train_bad_boys)(alpha_eig=eig, alpha_jacob=jacob,
                                                                                eps=ep, cuda=cuda, nonlin=nonlin,
                                                                                arch='cnn', max_epochs=100,
                                                                                save_dir='../data/cnn_1/',
                                                                                seed=seed) for seed in seeds)


if __name__ == '__main__':
    fire.Fire(money)

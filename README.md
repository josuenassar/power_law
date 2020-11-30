# power_law
This is the accompanying repository for our NeurIPS 2020 paper titled "On 1/n neural representation and robustness". 

# Background

A recent finding by Stringer et al. 2019 found that eigenspectrum of the empirical covariance matrix of the neural activity in mouse V1 followed a power-law, i.e. 
<a href="https://www.codecogs.com/eqnedit.php?latex=\lambda_n&space;\propto&space;n^{-\alpha}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\lambda_n&space;\propto&space;n^{-\alpha}" title="\lambda_n \propto n^{-\alpha}" /></a> regardless of the input statistics, where for natural images a universal exponent of <a href="https://www.codecogs.com/eqnedit.php?latex=\alpha&space;\approx&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha&space;\approx&space;1" title="\alpha \approx 1" /></a> was observed. A corresponding theory was put forward as a potential rationale for the existence of a <a href="https://www.codecogs.com/eqnedit.php?latex=n^{-1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n^{-1}" title="n^{-1}" /></a> eigenspectrum. While the theory was illuminating, the advantages of a representation with eigenspectrum decaying slightly faster than <a href="https://www.codecogs.com/eqnedit.php?latex=n^{-1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n^{-1}" title="n^{-1}" /></a> is not apparent. To investigate further, we turn to deep neural networks as a testbed.

# Spectrally regularized Deep Neural Networks
In general, the distribution of eigenvalues of a deep neural network is intractable and a priori there is no reason to believe it should follow a power law. To enforce a <a href="https://www.codecogs.com/eqnedit.php?latex=n^{-1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n^{-1}" title="n^{-1}" /></a> eigenspectrum, we directly regularize the eigenvalues of the activations at layer l, denoted by <a href="https://www.codecogs.com/eqnedit.php?latex=\lambda_n^l" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\lambda_n^l" title="\lambda_n^l" /></a>, towards a desired eigenspectrum, denoted by <a href="https://www.codecogs.com/eqnedit.php?latex=\gamma^l" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\gamma^l" title="\gamma^l" /></a>, which follows a <a href="https://www.codecogs.com/eqnedit.php?latex=n^{-1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n^{-1}" title="n^{-1}" /></a> power-law:

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\beta}{N_l}\sum_{n&space;\geq&space;\tau}^{N_l}\left(&space;(\lambda_n^l&space;/&space;\gamma_n^l&space;-&space;1)^2&space;&plus;&space;\max(0,&space;\lambda_n^l/\gamma_n^l&space;-&space;1)&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\beta}{N_l}\sum_{n&space;\geq&space;\tau}^{N_l}\left(&space;(\lambda_n^l&space;/&space;\gamma_n^l&space;-&space;1)^2&space;&plus;&space;\max(0,&space;\lambda_n^l/\gamma_n^l&space;-&space;1)&space;\right&space;)" title="\frac{\beta}{N_l}\sum_{n \geq \tau}^{N_l}\left( (\lambda_n^l / \gamma_n^l - 1)^2 + \max(0, \lambda_n^l/\gamma_n^l - 1) \right )" /></a>

where \tau is a cut-off that dictates which eigenvalues should be regularized and \beta is a hyperparameter that controls the strength of the regularizer.

# Running Experiments

In `neurips_experiments/`, `experiment_1.py`, `experiment_2.py` and `experiment_3.py` correspond to the experiments ran in sections 4.1, 4.2 and 4.3 respectively.

## Experiment 1
The vanilla networks can be trained by calling `neurips_experiments/experiment_1.py --vanilla=True`. 

The spectrally regularized networks can be trained by calling `neurips_experiments/experiment_1.py --vanilla=False`. 

Note that the networks can be trained on the GPU by passing in `--cuda=True`.

## Experiment 2
The vanilla MLPs can be trained by calling `neurips_experiments/experiment_2.py --vanilla=True --arch="mlp" `.

The MLPs with whitened intermediate representations (denoted as Vanilla-Wh in section 4.2) can be trained by calling `neurips_experiments/experiment_2.py --vanilla=True --arch="mlp" --flat=True`.

The MLPs where only the last hidden layer is spectrally regularized (denoted as SpecReg in section 4.2) can be trained by calling `neurips_experiments/experiment_2.py --vanilla=False --arch="mlp" `.

The MLPs where only the last hidden layer is spectrally regularized and the intermediate representation is whitend (denoted as SpecReg-Wh in section 4.2) can be trained by calling `neurips_experiments/experiment_2.py --vanilla=False --arch="mlp" --flat=True `.

(Note that to repeat these same experiments on CNNs pass in `--arch="cnn"` instead. Also, networks can be trained on the GPU by passing in `--cuda=True`.)

## Experiment 3
The vanilla MLPs can be trained by calling `neurips_experiments/experiment_3.py --vanilla=True --arch="mlp" `.

The MLPs where every hidden layer is spectrally regularized (denoted as SpecReg in section 4.3) can be trained by calling `neurips_experiments/experiment_3.py --vanilla=False --arch="mlp" `.

The MLPs whose input-output Jacobian is regularized (denoted as Jac in section 4.3) can be trained by calling `neurips_experiments/experiment_3.py --vanilla=True --arch="mlp" --jac=True`.

(Note that to repeat these same experiments on CNNs pass in `--arch="cnn"` instead. Also, networks can be trained on the GPU by passing in `--cuda=True`.)

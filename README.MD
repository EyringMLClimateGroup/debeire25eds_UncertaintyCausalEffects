# Uncertainty bounds for long-term causal effects of perturbations in spatio-temporal systems

In this folder, you can find the official implementation of the numerical experiments of the paper:
> Debeire, K., Gerhardus, A., Bichler, R., Runge, J., & Eyring, V. (2024). Uncertainty bounds for long-term causal effects of perturbations in spatio-temporal systems.

First **setup a conda environment** (by default called bagged_pcmci) from the [environment.yml](./environment.yml) file:

```setup
conda env create -f environment.yml
```

**Activate** this environment.

The PCMCI method is implemented in the [TIGRAMITE](https://github.com/jakobrunge/tigramite/) package. Follow the instructions below to install TIGRAMITE.

First clone the TIGRAMITE repository:

```
git clone https://github.com/jakobrunge/tigramite.git
```

Then install TIGRAMITE:

```
python setup.py install
```

 You should now be able to run the numerical experiments and reproduce the figures of the paper.
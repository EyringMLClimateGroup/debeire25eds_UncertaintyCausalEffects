# Uncertainty bounds for long-term causal effects of perturbations in spatio-temporal systems

In this folder, you can find the official implementation of the numerical experiments and application of the paper:
> Debeire, K., Gerhardus, A., Bichler, R., Runge, J., & Eyring, V. (2025). Uncertainty bounds for long-term causal effects of perturbations in spatio-temporal systems.
[![DOI](https://zenodo.org/badge/957902120.svg)](https://zenodo.org/badge/latestdoi/957902120)

## Requirements

First **setup a conda environment** (by default called lte_env) from the [environment.yml](./environment.yml) file:

```setup
conda env create -f environment.yml
```

**Activate** this environment.

```
conda activate lte_env
```

We use the [SAVAR](https://github.com/xtibau/savar.git) package to generate the synthetic SAVAR data. Follow the instructions to install the SAVAR package.

```
pip install git+https://github.com/xtibau/savar.git#egg=savar
```

The PCMCI method is implemented in the [TIGRAMITE](https://github.com/jakobrunge/tigramite/) package. Follow the instructions below to install TIGRAMITE. 

First clone the TIGRAMITE repository:

```
git clone https://github.com/jakobrunge/tigramite.git
```

Then point to this specific commit for consistency:

```
git reset --hard e9fe9db30a22d13761420be3c8f296306bc63534
```

Then install TIGRAMITE:

```
python setup.py install
```

 You should now be able to run the numerical experiments and reproduce the figures of the paper.

## Generating numerical experiments data and plotting figures

Find below the instructions to produce the **figures** of the main text and of the appendix for the two numerical experiments.

**Note**: you can modify and provide new methods for Dimension reduction, Causal discovery and Linear coefficient estimation that you want to benchmark to estimate sensitivity and long-run effects. For this purpose, please add the desired methods in the appropriate python file *DimensionReduction.py*, *CausalDiscovery.py* or *LinCoeffEstimation.py* as a subclass of the main class. Take care that you provide the required attributes and methods. The benchmark will then automatically include your new subclass. All combination of DimensionReduction, Causal Discovery and LinearCoefficient will be tested during the benchmark.

### First numerical experiment: generating data and benchmarking

To generate the synthetic data and to benchmark the methods:
- run: 'python run_benchmark_sens_err_L.py'
- run: 'python run_benchmark_sens_err_T.py'
- run: 'python run_benchmark_sens_err_auto.py'
- run: 'python run_benchmark_sens_err_cross.py'
- run: 'python run_benchmark_sens_err_lambda.py'
- and finally 'python run_benchmark_sens_err_modes.py'

### First numerical experiment: plotting benchmark figures
Once the synthetic data have been generated and saved in the *./output/* folder :
- you can produce the figures in the [plot_benchmark_sens_err.ipynb](./plot_benchmark_sens_err.ipynb) Jupyter notebook.
- the figures are saved in './plots/benchmark/sens_err/' that you may need to create

### Second numerical experiment: generating synthetic data and computation of Confidence intervals 

The different methods to build Confidence Intervals are benchmarked with synthetic data generated:
- to produce the average confidence interval sizes over different SAVAR models use 'sbatch sbatch_mpi_allseeds.sh' which will embarassingly parallelize the bootstrap-based methods.
- to produce the confidence interval sizes for one SAVAR model (associated with a seed), use 'sbatch sbatch_mpi_1seed.sh' which will embarassingly parallelize the bootstrap-based methods.

### Second numerical experiment: plotting figures
The results of the previous stage are saved in *./output/*. To display the figure which will be saved in *./plots/ci/*:
- run the [plot_confidence_interval_allseeds.ipynb](./plot_confidence_interval_allseeds.ipynb) Jupyter notebook for the experiment with  average confidence interval sizes over different SAVAR models.
- run the [plot_confidence_interval_1seed.ipynb](./plot_confidence_interval_1seed.ipynb)Jupyter notebook for the experiment with confidence interval sizes for one SAVAR model (associated with a seed).

## Real-world application: Air pollution in Northern Italy.

For this experiment, we provide the following curated dataset:
- economic time series in *./data/eco_data/* 
- air pollution data in *./data/Po_valley_results/*

The analysis and figures of the paper can be reproduced by running the [plot_results_NO2_application.ipynb](./plot_results_NO2_application.ipynb) Jupyter notebook.

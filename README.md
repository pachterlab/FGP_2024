# Introduction
This repository contains code for reproducing the figures and results in the paper [Meichen Fang, Gennady Gorin, Lior Pachter (2025)](https://doi.org/10.1371/journal.pcbi.1012752). 

# Repository Contents
 
* `Chronocell/`: A Python package that implements the fitting procedures. The core code for the package is maintained in a separate repository https://github.com/pachterlab/Chronocell. A copy is included here for completeness. 
  * `inference.py`: contains the `Trajectory` class and methods for fitting.
  * `mixtures.py`: contains the code and classes for Poisson mixture model.
  * `models/`: contains the model specific functions to calculate log likelihood and optimize parameters.
  * `simulation.py`: contains the code for generating simulations.
  * `plotting.py`: contains some convenient but not essential functions for plotting.
  * `utils.py`: contains some helper functions.

* `simulations/`: notebooks to perform analyses on simulations.

* `notebooks/`: notebooks to perform analyses on real datasets.
  * `forebrain.ipynb`
  * `erythroid.ipynb`
  * `rpe1.ipynb`
  * `PBMC.ipynb`
  * `Neuron.ipynb`

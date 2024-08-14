# Introduction
This repository contains code for reproducing the figures and results in the preprint [Meichen Fang, Gennady Gorin, Lior Pachter (2024)]([https:](https://www.biorxiv.org/content/10.1101/2024.01.26.577510v2)).

# Repository Contents
 
* `Chronocell/`: a python package that implement the fitting
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

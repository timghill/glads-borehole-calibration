# GlaDS borehole calibration


Tim Hill, 2025 (tim_hill_2@sfu.ca)
https://github.com/timghill/glads-borehole-calibration

Code corresponding to "Emulator-based Bayesian calibration of a subglacial drainage model". This project uses a Gaussian Process emulator and borehole hydraulic head measurements to calibrate 8 uncertain parameters of the Glacier Drainage System (GlaDS) model ([Werder et al., 2013](https://doi.org/10.1002/jgrf.20146)). This repository consists of code to run the GlaDS simulation ensembles, run the MCMC sampling to infer the posterior probability distributions, and make figures for the published paper.

## Description

The project structure is:

 - `issm/`: ISSM-GlaDS simulation directories
 - `data/`: Processed daily borehole flotation-fraction measurements
 - `analysis/`: Calibration experiments sub-directories
 - `utils/`: Shared code for consistent simulation settings and MCMC sampling

Each directory has its own `README.md` file that describes the contents in more detail.


## Requirements

Running these scripts requires python 3. The analysis source code has been tested against python 3.12.4. Package requirements are listed in `requirements.txt`, and it is recommended to use a virtual environment to manage versions. For example

```
virtualenv --python 3.12.4 pyenv/
source pyenv/bin/activate
pip install -r requirements.txt
```

Running ISSM requires a working installation of [ISSM v4.24](https://github.com/ISSMteam/ISSM/releases/tag/v4.24).

This code depends on a fork of the SEPIA package ([timghill/SEPIA](https://github.com/timghill/SEPIA)) that that can be installed using `pip install -e .`.

## Installation

The source code in `utils/` is installed as an editable python package. To install this code, use pip with the `-e` option:

```
pip install -e .
```

Your python environment and installation can be verified by running `test_install.sh`. This script should run with no errors and should update two figures in `experiments/synthetic/analysis/figures/`.


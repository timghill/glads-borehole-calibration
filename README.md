# Bayesian calibration of subglacial drainage model using borehole water pressure timeseries data

Tim Hill, 2024 (tim_hill_2@sfu.ca)
https://github.com/timghill/GladsGP/blob/main/README.md


Code corresponding to "Emulator-based Bayesian calibration of a subglacial drainage model".

This project calibrates parameters of the Glacier Drainage System (GlaDS) model ([Werder et al., 2013](https://doi.org/10.1002/jgrf.20146)) with borehole water-pressure timeseries ([Meierbachtol et al.,  2013](https://doi.org/10.1126/science.1235905); [Wright et al., 2016](https://doi.org/10.1002/2016JF003819)) using Gaussian process-based Bayesian inference.

## Description

The project structure is:

 * `utils/`: shared code for setting up and running ISSM simulations and for analyzing outputs
 * `issm/`: GlaDS-ISSM simulation directories
 * `analysis/`: individual directories for analyzing model experiments

Each directory has a README file to describe the contents. Original datasets used as inputs for ISSM and the original borehole dataset are not included in this repository, see the individual `citation.md` files.

## Installation

The analysis source code has been tested against python 3.12.4. Package requirements are listed in `requirements.txt`, and it is recommended to use a virtual environment to manage versions. For example

```
virtualenv --python 3.12 pyenv/
source pyenv/bin/activate
pip install -r requirements.txt
```

To install the code for this project on your python path, install in editable (`-e`) mode with pip:

```
pip install -e .
```

This code also depends on a fork of the SEPIA package ([timghill/SEPIA](https://github.com/timghill/SEPIA)) that can be installed using `pip install -e .`, and simulations are run with the Ice-sheet and Sea-level System Model ([ISSM](https://github.com/ISSMteam/ISSM)).

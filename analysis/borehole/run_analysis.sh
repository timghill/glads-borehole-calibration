#!/bin/bash

train='../../train_config.py'
bh='../../GL12-2A.pkl'
burn=500

set -x

# Evaluate sensitivity to number of PCs and training runs
python -u compare_models.py $train $bh --burn $burn

# Separate calibration experiment
python -u plot_separate_calibration.py $train data/model_m512_p12.pkl data/summer_m512_p12.pkl data/winter_m512.pkl --burn 500

# Posterior predictions
python -u plot_posterior.py $train ../../post_borehole_config.py $bh
python -u compare_posterior.py $train ../../post_borehole_config.py ../../post_borehole_config.py $bh

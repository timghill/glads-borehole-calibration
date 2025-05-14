#!/bin/bash

# Test the python environment by running analysis scripts

set -x

cd analysis/synthetic

train='../../train_config.py'
test='../../test_config.py'
bh='../../GL12-2A.pkl'

python -u plot_posterior.py $train $test ../../post_synthetic_config.py $bh

#!/bin/bash

train='../../train_config.py'
test='../../test_config.py'
bh='../../GL12-2A.pkl'
burn=500

set -x

# Evaluate PC truncation error
python -u eval_pcs.py $train $test $bh

# Evaluate sensitivity to number of PCs and training runs
python -u compare_models.py $train $test $bh --burn $burn

# Assess GP performance for specified m, p
python -u eval_gp.py $train $test $bh --m 512 --p 15

# Look at different nodes
python -u node_choice.py $train $bh

# Posterior predictions
python -u plot_posterior.py $train $test ../../post_synthetic_config.py $bh

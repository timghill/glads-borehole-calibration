#!/bin/bash
#SBATCH --time=0-01:00
#SBATCH --mem=8G
#SBATCH --account=def-gflowers
#SBATCH --mail-user=tha111@sfu.ca
#SBATCH --mail-type=FAIL,END,BEGIN

source ~/bhcal-pyenv/bin/activate

python compare_posterior.py ../../train_config.py ../../post_synthetic_config.py ../../post_synthetic_config.py ../../GL12-2A.pkl


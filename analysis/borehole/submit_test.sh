#!/bin/bash
#SBATCH --job-name=SynthTest
#SBATCH --time=0-06:00
#SBATCH --account=def-gflowers
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --mail-user=tha111@sfu.ca
#SBATCH --mail-type=FAIL,END

module load python/3.12.4
source ../../bhcal-pyenv/bin/activate

#python -u test_emulator.py ../../train_config.py ../../test_config.py data/ypred_test.npy --recompute

python -u test_GP.py ../../train_config.py ../../test_config.py data/model

#!/bin/bash
#SBATCH --time=0-02:00
#SBATCH --account=def-gflowers
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=aggregate.out
#SBATCH --error=aggregate.err

source ../setenv.sh

python -u -m utils.aggregate_outputs ../../../train_config.py 1024


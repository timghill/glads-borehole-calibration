#!/bin/bash
#SBATCH --time=0-00:30
#SBATCH --account=def-gflowers
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --output=aggregate.out
#SBATCH --error=aggregate.err

source ../setenv.sh

python -u -m utils.aggregate_outputs ../../../test_config.py 100


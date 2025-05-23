#!/bin/bash
#SBATCH --time=0-01:00
#SBATCH --account=def-gflowers
#SBATCH --mem=24G
#SBATCH --ntasks=1
#SBATCH --output=aggregate.out
#SBATCH --error=aggregate.err
#SBATCH --job-name=collectBH

source ../setenv.sh

python -u -m utils.aggregate_outputs ../../../post_borehole_config.py 256


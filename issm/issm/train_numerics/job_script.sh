#!/bin/bash
#SBATCH --job-name="gr-train"
#SBATCH --time=0-64:00:00
#SBATCH --mem=2G
#SBATCH --account=def-gflowers
#SBATCH --mail-user=tha111@sfu.ca
#SBATCH --mail-type=FAIL,END

# 4 hours/simulation

source ../setenv.sh
task.run

#!/bin/bash
#SBATCH --job-name="SynthFit"
#SBATCH --time=0-16:00:00
#SBATCH --mem=2G
#SBATCH --account=def-gflowers
#SBATCH --mail-user=tha111@sfu.ca
#SBATCH --mail-type=FAIL,END

module load python/3.12.4
source ~/bhcal-pyenv/bin/activate

task.run

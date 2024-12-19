#!/bin/bash
#SBATCH --job-name="SynthFit"
#SBATCH --time=0-24:00:00
#SBATCH --mem=4G
#SBATCH --account=def-gflowers
#SBATCH --mail-user=tha111@sfu.ca
#SBATCH --mail-type=FAIL,END,BEGIN

module load python/3.12.4
source ~/bhcal-pyenv/bin/activate

task.run

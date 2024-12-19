#!/bin/bash
#SBATCH --job-name=summer
#SBATCH --time=0-16:00:00
#SBATCH --account=def-gflowers
#SBATCH --mem=2G
#SBATCH --ntasks=1
#SBATCH --output=logs/summer.out
#SBATCH --error=logs/summer.err
#SBATCH --mail-user=tha111@sfu.ca
#SBATCH --mail-type=FAIL,END,BEGIN

module load python/3.12.4
source ~/bhcal-pyenv/bin/activate

python -u fit_summer.py train_config.py ../../GL12-2A.pkl --p 12 --m 512 --sample 5000 --burn 500 --recompute

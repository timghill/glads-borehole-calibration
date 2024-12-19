#!/bin/bash
#SBATCH --job-name=winter
#SBATCH --time=0-02:00:00
#SBATCH --account=def-gflowers
#SBATCH --mem=2G
#SBATCH --ntasks=1
#SBATCH --output=logs/winter.out
#SBATCH --error=logs/winter.err
#SBATCH --mail-user=tha111@sfu.ca
#SBATCH --mail-type=FAIL,END,BEGIN

module load python/3.12.4
source ../../bhcal-pyenv/bin/activate

python -u fit_winter.py train_config.py ../../GL12-2A.pkl --m 512 --sample 5000 --burn 500 --recompute

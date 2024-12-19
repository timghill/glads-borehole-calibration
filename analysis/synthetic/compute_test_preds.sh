#!/bin/bash
#SBATCH --job-name=SynthTestPreds
#SBATCH --time=0-04:00
#SBATCH --account=def-gflowers
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --mail-user=tha111@sfu.ca
#SBATCH --mail-type=FAIL,END,BEGIN
#

module load python/3.12.4
source ~/bhcal-pyenv/bin/activate

python -u compute_test_preds.py ../../train_config.py ../../test_config.py ../../GL12-2A.pkl --sample 256 --burn 500
python -u eval_gp.py ../../train_config.py ../../test_config.py ../../GL12-2A.pkl --m 512 --p 15

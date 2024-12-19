#!/bin/bash
#SBATCH --job-name=SynthFit
#SBATCH --time=0-06:00
#SBATCH --account=def-gflowers
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --output=logs/fit-all.out
#SBATCH --error=logs/fit-all.err
#SBATCH --mail-user=tha111@sfu.ca
#SBATCH --mail-type=FAIL,END

module load python/3.12.4
source ../../bhcal-pyenv/bin/activate

python -u fit_model.py ../../train_config.py ../../test_config.py --p 5 --m 128 --sample 50 --burn 0 --recompute
python -u -m utils.trace_plot ../../train_config.py data/model_m128_p05.pkl --burn 0 --save trace_m128_p05_{}.png


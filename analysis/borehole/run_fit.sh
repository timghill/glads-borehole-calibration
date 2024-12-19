#!/bin/bash

printf -v pstr "%02d" $1
printf -v mstr "%03d" $2

python -u fit_model.py train_config.py ../../GL12-2A.pkl --p $1 --m $2 --sample 5000 --burn 500 --recompute
python -u -m utils.trace_plot train_config.py data/model_m${mstr}_p${pstr}.pkl --burn 0 --save m${mstr}_p${pstr}/trace_m${mstr}_p${pstr}_{}.png

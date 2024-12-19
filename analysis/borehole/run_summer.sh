#!/bin/bash

printf -v pstr "%02d" $1
printf -v mstr "%03d" $2

python -u fit_summer.py ../../train_config.py ../../GL12-2A.pkl --p $1 --m $2 --sample 250 --burn 50 --recompute
python -u -m utils.trace_plot ../../train_config.py data/summer_m${mstr}_p${pstr}.pkl --burn 0 --save summer/m${mstr}_p${pstr}/trace_m${mstr}_p${pstr}_{}.png


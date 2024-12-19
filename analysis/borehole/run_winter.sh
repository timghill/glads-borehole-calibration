#!/bin/bash

printf -v mstr "%03d" $1

python -u fit_winter.py ../../train_config.py ../../GL12-2A.pkl --m $1 --sample 500 --burn 100 --recompute
python -u -m utils.trace_plot ../../train_config.py data/winter_m${mstr}.pkl --burn 0 --save winter/m_${mstr}/trace_m${mstr}_{}.png

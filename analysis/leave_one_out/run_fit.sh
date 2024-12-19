#!/bin/bash

printf -v istr "%03d" $1

python -u fit_model.py ../../train_config.py ../../test_config.py ../../GL12-2A.pkl $1 --p 15 --m 512 --sample 2500 --burn 500 --recompute


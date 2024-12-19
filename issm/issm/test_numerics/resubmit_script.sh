#!/bin/bash
#SBATCH --job-name="Seas resubmit"
#SBATCH --time=00-48:0:00
#SBATCH --mem=8G
#SBATCH --account=def-gflowers
#SBATCH --mail-user=tha111@sfu.ca
#SBATCH --mail-type=FAIL,END



# Don't change this line:
autojob.run

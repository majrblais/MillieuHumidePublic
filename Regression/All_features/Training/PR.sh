#!/bin/bash
#SBATCH --account=def-akhloufi
#SBATCH --cpus-per-task=1

#SBATCH --mem=4G
#SBATCH --time=0-11:40:00

cd /home/emb9357/
module load cuda
source tensorflow/bin/activate
cd scratch/Millieu_Humides/Regression/All_features/Training_norm


nohup python -u PR.py 

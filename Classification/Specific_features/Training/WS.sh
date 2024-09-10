#!/bin/bash
#SBATCH --account=def-akhloufi
#SBATCH --cpus-per-task=1

#SBATCH --mem=4G
#SBATCH --time=0-00:40:00

cd /home/emb9357/
module load cuda
source tensorflow/bin/activate
cd scratch/Millieu_Humides/Classification/Specific_features/Training/


nohup python -u WS.py

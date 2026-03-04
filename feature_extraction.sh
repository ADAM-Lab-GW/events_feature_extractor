#!/bin/bash

#SBATCH -o LOG_%j.out
#SBATCH -e LOG_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH -D /scratch/adamgrp/repos/events_feature_extractor
#SBATCH -J feature_extraction
#SBATCH --export=NONE
#SBATCH -t 7-00:00:00
# this is how is get an entire node: SBATCH --exclusive

module purge
module load python3/3.9.2
module load gcc/12.2.0
source venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1

#python3 create_eventSym_dataset.py 
python3 train_feature_extractor.py
#python3 extract_features.py

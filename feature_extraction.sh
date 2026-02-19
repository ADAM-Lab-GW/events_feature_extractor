#!/bin/bash

#SBATCH -o LOG_%j.out
#SBATCH -e LOG_%j.err
#SBATCH -p med-gpu
#SBATCH -N 1
#SBATCH -D /scratch/adamgrp/repos/events_feature_extractor
#SBATCH -J feature_extraction
#SBATCH --export=NONE
#SBATCH -t 7-00:00:00
#SBATCH --nice=100

source venv/bin/activate
python3 create_eventSym_dataset.py 
python3 train_feature_extractor.py
python3 extract_features.py

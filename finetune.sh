#!/bin/bash

cd /home/santosh/Projects/geo_physics/JOURNAL_RUNS/pretraining/

source /home/santosh/miniconda3/etc/profile.d/conda.sh
conda activate ocr
CUDA_VISIBLE_DEVICES=1 python train.py --config ./configs/finetune.yaml
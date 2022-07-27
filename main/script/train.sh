#!/bin/bash

export PYTHONPATH="./main/python"
MODEL_PRETRAINED='PlanTL-GOB-ES/bsc-bio-es'
# NUMBER_OF_STEPS=5000 #Parameter moved to config.py
# CUDA_ID=1 #Parameter moved to config.py
WANDB_PROJECT_NAME="PlanTL-GOB-ES/bsc-bio-es" #Not used right now

nohup python -u ./main/python/disease_classifier/transformers/disease_ner.py ${MODEL_PRETRAINED} $WANDB_PROJECT_NAME >> ./main/logs/${MODEL_PRETRAINED}.log &
#nohup python -u ./main/python/disease_classifier/multitrainer_script.py >>  ./main/logs/multi_2.log &

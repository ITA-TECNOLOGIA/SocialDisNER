#!/bin/bash
export PYTHONPATH="./main/python"

nohup python -u ./main/python/main_predict_validation.py  >> ./main/logs/predict_validation.log &

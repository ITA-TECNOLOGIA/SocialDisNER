#!/bin/bash
export PYTHONPATH="./main/python"

nohup python -u ./main/python/main_predict_test.py  >> ./main/logs/predict_test.log &

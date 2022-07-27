#!/bin/bash
export PYTHONPATH="./main/python"

nohup python -u ./main/python/main_filter_dictionary.py  >> ./main/logs/filter_dictionary.log &

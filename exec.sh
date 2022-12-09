#!/bin/bash
tensorboard --port 8080 --logdir . serve &
PID=$!
pip install --user -r requirements.txt
python -u main.py
kill $PID

#!/bin/bash
tensorboard --port 8080 --logdir . serve &
PID=$!
python -u main.py
kill $PID

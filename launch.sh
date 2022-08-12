#!/bin/bash
kubectl delete job mnist
kubectl create -f job.yaml

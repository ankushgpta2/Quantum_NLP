#!/bin/bash
kubectl delete job quantumnlp
kubectl create -f job.yaml

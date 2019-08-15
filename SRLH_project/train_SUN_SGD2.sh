#!/bin/bash

lambda_1s=(0.01 0.02 0.03 0.04)
lambda_2s=(0.01 0.02 0.03 0.04)
lambda_3s=(0)

echo 'SRLH_ndcg_100_Adam' 

python run_SRLH_ori3.py --optimizer=Adam --epochs=50 --batch-size=100 --learning-rate=0.0001 --gamma=0.5 --bits=64 --num-samples=1000 --dataname=SUN20 --gpu 0,1


#!/bin/bash

lambda_1s=(0.01 0.02 0.03 0.04)
lambda_2s=(0.01 0.02 0.03 0.04)
lambda_3s=(0)

echo 'SRLH_ndcg_100_RMSprop_ori6' 

python run_SRLH_ori7.py --optimizer=RMSprop --sparsity=10 --EPSILON=0.1 --epochs=50 --batch-size=100 --learning-rate=0.002 --gamma=0.6 --bits=64 --num-samples=1000 --dataname=SUN20 --gpu=6,7


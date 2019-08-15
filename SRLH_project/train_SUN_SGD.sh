#!/bin/bash

lambda_1s=(0.01 0.02 0.03 0.04)
lambda_2s=(0.01 0.02 0.03 0.04)
lambda_3s=(0)

echo 'SRLH_ndcg_100_SGD'

#python run_SRLH_ori.py --epochs=20 --batch-size=48 --learning-rate=0.0001 --gamma=0.2 --bits=64 --num-samples=1000 --dataname=SUN20 --gpu 0

python run_SRLH_ori2.py --epochs=20 --batch-size=64 --learning-rate=0.00001 --gamma=0.2 --bits=64 --num-samples=1000 --dataname=SUN20 --gpu 1

#python run_SRLH_new_reward.py --epochs=20 --batch-size=48 --learning-rate=0.0001 --gamma=0.9 --bits=64 --num-samples=1000 --dataname=SUN20 --gpu 0

#python run_SRLH_adam.py --epochs=10 --learning-rate=0.00001 --gamma=0.9 --bits=64 --num-samples=1000 --dataname=SUN20 --gpu 2


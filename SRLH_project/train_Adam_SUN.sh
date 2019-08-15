#!/bin/bash

lambda_1s=(0.01 0.02 0.03 0.04)
lambda_2s=(0.01 0.02 0.03 0.04)
lambda_3s=(0)

echo 'SRLH'


python run_SRLH_new_reward.py --arch=resnet50 --epochs=20 --batch-size=64 --learning-rate=0.0001 --gamma=0.9 --bits=64 --num-samples=1000 --dataname=SUN20 --gpu 1


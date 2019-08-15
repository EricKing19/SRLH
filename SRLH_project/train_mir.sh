#!/bin/bash

lambda_1s=(0.01 0.02 0.03 0.04)
lambda_2s=(0.01 0.02 0.03 0.04)
lambda_3s=(0)

echo 'SRLH_MirFlickr'

python run_SRLH_adam.py --epochs=10 --batch-size=80 --learning-rate=0.00001 --gamma=0.2 --num-samples=1000 --bits=64 --dataname=MirFlickr --gpu 0




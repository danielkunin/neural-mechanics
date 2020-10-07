#!/bin/bash

python train.py --overwrite --experiment=cifar100 --expid=tpu-vgg16-bn --verbose --save-dir=gs://jvr-pt-tpu --lr=0.01 --wd=5e-4 --dataset=cifar100 --data-dir=/home/jvrsgsty/data --model=vgg16-bn --model-class=tinyimagenet --train-batch-size=32 --epochs=10 --save-freq=100 --tpu=jv-pt-tpu-01


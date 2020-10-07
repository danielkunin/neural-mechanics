#!/bin/bash

python train.py --overwrite --experiment=cifar10 --expid=tpu-vgg16 --verbose --save-dir=gs://jvr-pt-tpu --lr=0.01 --dataset=cifar10 --data-dir=/home/jvrsgsty/data --model=vgg16 --model-class=tinyimagenet --train-batch-size=32 --epochs=10 --save-freq=100 --tpu=jv-pt-tpu-01


#!/bin/bash

python train.py --overwrite --experiment=imagenet --expid=tpu-vgg16 --verbose --save-dir=gs://jvr-pt-tpu --lr=0.01 --wd=5e-4 --dataset=imagenet --data-dir=/home/jvrsgsty/data --model=vgg16 --model-class=imagenet --train-batch-size=32 --epochs=1 --save-freq=100 --tpu=jv-pt-tpu-01


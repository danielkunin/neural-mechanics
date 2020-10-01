#!/bin/bash

# TPU training with pytorch: sets necessary env variables

export TPU_IP_ADDRESS=10.8.170.2
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"

python train.py --overwrite --experiment=neural-mechanics --expid=vgg16-bn_bs256_lr1en2_wd0 --verbose --save-dir=gs://jvr-pt-tpu --dataset=imagenet --data-dir=/home/jvrsgsty/data --model=vgg16-bn --model-class=imagenet --train-batch-size=32 --lr=0.01 --wd=0 --optimizer=custom_sgd --epochs=100 --save-freq=500 --tpu=True

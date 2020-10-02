#!/bin/bash

# TPU training with pytorch: sets necessary env variables

export TPU_IP_ADDRESS=10.3.170.2
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"

python train.py --overwrite --experiment=tinyimagenet --expid=vgg16_bs256_lr1en1_wd1en3 --verbose --save-dir=gs://jvr-pt-tpu --dataset=tiny-imagenet --data-dir=/home/jvrsgsty/data --model=vgg16 --model-class=tinyimagenet --train-batch-size=32 --lr=0.1 --wd=1e-3 --optimizer=custom_sgd --epochs=100 --save-freq=500 --tpu=True --workers=8

#!/bin/bash

# TPU training with pytorch: sets necessary env variables

export TPU_IP_ADDRESS=10.2.170.2
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"

python train.py --save-dir=gs://jvr-pt-tpu --experiment=fo-vgg16 --expid=lr1en1_wd1en4_bn --lr=0.1 --wd=0.0001 --optimizer=custom_sgd --train-batch-size=16 --epochs=100 --seed=1 --data-dir=/home/jvrsgsty/data --dataset=cifar100 --model-class=tinyimagenet --model=vgg16-bn --save-freq=250 --tpu=True --overwrite --workers=8

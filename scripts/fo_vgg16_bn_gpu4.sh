#!/bin/bash

# train
for lr in 0.1; do
    for wd in 0.0 0.01 0.001 0.0001; do
        python train.py --save-dir=/mnt/fs6/jvrsgsty/neural-mechanics --experiment=vgg16-bn --expid=lr_${lr}_wd_${wd} --lr=$lr --wd=$wd --optimizer=custom_sgd --train-batch-size=128 --epochs=100 --seed=1 --data-dir=/mnt/fs6/jvrsgsty/data --dataset=cifar100 --model-class=tinyimagenet --model=vgg16-bn --save-freq=250 --overwrite --gpu=4
    done
done

# extract
for lr in 0.1; do
    for wd in 0.0 0.01 0.001 0.0001; do
        python extract.py --save-dir=/mnt/fs6/jvrsgsty/neural-mechanics --experiment=vgg16-bn --expid=lr_${lr}_wd_${wd} --overwrite --gpu=4
    done
done

# plot
for lr in 0.1; do
    for wd in 0.0 0.01 0.001 0.0001; do
        python visualizations/scale.py --save-dir=/mnt/fs6/jvrsgsty/neural-mechanics --experiment=vgg16-bn --expid=lr_${lr}_wd_${wd} --overwrite
    done
done

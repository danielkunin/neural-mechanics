#!/bin/bash

# train
for lr in 0.1; do
  for wd in 0.0005 0.001; do
    for beta in 0.8 0.9 0.99; do
      for gamma in 0 0.01 0.01; do
        python train.py --save-dir=/mnt/fs6/jvrsgsty/neural-mechanics --experiment=vgg16-bn --expid=timnet_momentum_lr_${lr}_wd_${wd}_beta_${beta}_gamma_${gamma} --lr=$lr --wd=$wd --momentum=$beta --dampening=$gamma --optimizer=custom_momentum --train-batch-size=128 --epochs=100 --seed=1 --data-dir=/mnt/fs6/jvrsgsty/data --dataset=tiny-imagenet --model-class=tinyimagenet --model=vgg16-bn --save-freq=250 --overwrite --gpu=1
      done
    done
  done
done

# extract
for lr in 0.1; do
  for wd in 0.0005 0.001; do
    for beta in 0.8 0.9 0.99; do
      for gamma in 0 0.01 0.01; do
        python extract-momentum.py --save-dir=/mnt/fs6/jvrsgsty/neural-mechanics --experiment=vgg16-bn --expid=timnet_momentum_lr_${lr}_wd_${wd}_beta_${beta}_gamma_${gamma} --overwrite --gpu=1
      done
    done
  done
done

# plot
for lr in 0.1; do
  for wd in 0.0005 0.001; do
    for beta in 0.8 0.9 0.99; do
      for gamma in 0 0.01 0.01; do
        python visualizations/translation-momentum.py --save-dir=/mnt/fs6/jvrsgsty/neural-mechanics --experiment=vgg16-bn --expid=timnet_momentum_lr_${lr}_wd_${wd}_beta_${beta}_gamma_${gamma} --overwrite
        python visualizations/scale-momentum.py --save-dir=/mnt/fs6/jvrsgsty/neural-mechanics --experiment=vgg16-bn --expid=timnet_momentum_lr_${lr}_wd_${wd}_beta_${beta}_gamma_${gamma} --overwrite
      done
    done
  done
done

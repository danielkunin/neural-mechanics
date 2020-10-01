
#!/bin/bash

# TPU training with pytorch: sets necessary env variables

tpu_name=$1
tpu_zone=$2

#gcloud compute tpus list --zone=$tpu_zone | grep $tpu_name

#export TPU_IP_ADDRESS=10.144.244.18
#export TPU_IP_ADDRESS=10.140.146.146
export TPU_IP_ADDRESS=10.9.170.2
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"

python train.py --overwrite --experiment=cifar10 --expid=tpu-vgg16 --verbose --save-dir=gs://jvr-pt-tpu --lr=0.01 --dataset=cifar10 --data-dir=/home/jvrsgsty/data --model=vgg16 --model-class=tinyimagenet --train-batch-size=32 --epochs=10 --save-freq=100 --tpu=True



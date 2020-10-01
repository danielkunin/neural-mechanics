
#!/bin/bash

# TPU training with pytorch: sets necessary env variables

tpu_name=$1
tpu_zone=$2

#gcloud compute tpus list --zone=$tpu_zone | grep $tpu_name

export TPU_IP_ADDRESS=10.2.170.2
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"

python train.py --overwrite --experiment=imagenet --expid=tpu-vgg16 --verbose --save-dir=gs://jvr-pt-tpu --lr=0.01 --wd=5e-4 --dataset=imagenet --data-dir=/home/jvrsgsty/data --model=vgg16 --model-class=imagenet --train-batch-size=32 --epochs=1 --save-freq=100 --tpu=True




#!/bin/bash

# TPU training with pytorch: sets necessary env variables

tpu_name=$1
tpu_zone=$2

#gcloud compute tpus list --zone=$tpu_zone | grep $tpu_name

export TPU_IP_ADDRESS=10.144.244.18
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"

python train.py --overwrite --experiment=tpu-test --expid=test2 --verbose --save-dir=~/Results --model=fc --train-batch-size=256 --epochs=10 --save-freq=20 --tpu=True



#!/bin/bash
zone=$3
version="pytorch-1.6"
tpu_type="v3-8"
for i in `seq $1 $2`; do
    if [ -z $zone ]
    then
        zone="us-central1-b"
    fi
    if [ -z $4 ]
    then
        printf -v tpu_name "jv-pt-tpu-%02d" $i
        address="10.$(($i)).170.0/29"
        echo "attempting to create $tpu_type $tpu_name at $address"
        gcloud compute tpus create $tpu_name --accelerator-type=$tpu_type --network=default --range=$address --version=$version --zone=$zone&
    else
        tpu_name="jv-pre-$i"
        address="10.$(($i)).232.0/29"
        echo "attempting to create preemtible v3 $tpu_name at $address"
        gcloud compute tpus create $tpu_name --accelerator-type=$tpu_type --preemptible --network=default --range=$address --version=$version --zone=$zone&
    fi
done

wait
echo "all tpus created successfully"

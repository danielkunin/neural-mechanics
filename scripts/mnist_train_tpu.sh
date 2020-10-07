#!/bin/bash

python train.py --overwrite --experiment=mnist --expid=tpu-test --verbose --save-dir=gs://jvr-pt-tpu --lr=0.01 --data-dir=/home/jvrsgsty/data --model=fc --train-batch-size=32 --epochs=10 --save-freq=20 --tpu=jv-pt-tpu-01 --workers=8


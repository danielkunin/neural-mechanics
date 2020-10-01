
#!/bin/bash


export TPU_IP_ADDRESS=10.144.244.18
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"

python /usr/share/torch-xla-1.5/pytorch/xla/test/test_train_mp_imagenet.py --fake_data --model=resnet50 --num_epochs=2 --batch_size=128 --log_steps=20


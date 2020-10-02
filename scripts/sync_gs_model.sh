#!/bin/sh

dest=/mnt/fs6/jvrsgsty/neural-mechanics/gs_jvr-pt-tpu
gsutil -m rsync -r gs://jvr-pt-tpu $dest


#!/bin/bash
pid_list=`ps aux | grep "train.py" | grep "jvrsgsty" | awk '{print $2}'`
for pid in $pid_list; do
    echo "Killing ${pid}"
    kill -9 $pid
done


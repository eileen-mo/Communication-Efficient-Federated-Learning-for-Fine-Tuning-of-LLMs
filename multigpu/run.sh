#!/bin/bash

if [[ $# -lt 1 ]]
then
    echo "specify number of clients: $0 numclients"
    return 1
fi
NUM_CLIENTS="$1"
ARGS="${@:2}"
NUM_GPUS=$(nvidia-smi -L | wc -l)

trap "pkill -P $$" SIGINT

echo "Starting server"
CUDA_VISIBLE_DEVICES=0 python3 server.py --num_clients $NUM_CLIENTS --rank 0 $ARGS  & 
sleep 3

for (( i=1; i<=$NUM_CLIENTS; i++ ))
do
    echo "Starting client $i"
    n=$(($i%$NUM_GPUS))
    CUDA_VISIBLE_DEVICES=$n python3 client.py --num_clients $NUM_CLIENTS --rank $i $ARGS &
done

wait

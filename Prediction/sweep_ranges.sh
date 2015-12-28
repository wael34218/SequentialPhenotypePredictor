#!/bin/bash

for i in `seq 250 25 300`; do
    echo "============ New window $i ============"
    for j in `seq 45 5 50`; do
        echo "Window $i - size $j"
        for k in `seq 7 1 8`; do
            echo "Decay - $k"
            python cbowsim.py --window $i --size $j --decay $k
        done
    done
done

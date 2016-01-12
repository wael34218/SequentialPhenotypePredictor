#!/bin/bash

for i in `seq 275 25 275`; do
    echo "============ New size $i ============"
    for j in `seq 45 5 50`; do
        echo "size $i - window $j"
        for k in `seq 9 1 13`; do
            echo "decay - $k"
            python3 cbowsim.py --size $i --window $j --decay $k
        done
    done
done

#!/bin/bash

for j in `seq 100 25 300`; do
    echo "============ New window Size $j ============"
    echo "============ New window Size $j ============" >> accuracy.txt
    for i in `seq 5 5 50`; do
        time ./../lib/word2vec -train ../Data/mimic_train_0 -output mimic-vectors.bin -cbow 0 -size $j -window $i -negative 25 -hs 0 -sample 1e-5 -threads 30 -binary 1 -iter 15
        echo "Window $j - Sequence $i" >> accuracy.txt
        python mimic-accuracy.py >> accuracy.txt
        echo "============" >> accuracy.txt
    done
done

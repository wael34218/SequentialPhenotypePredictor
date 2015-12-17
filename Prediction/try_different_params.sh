#!/bin/bash
for i in `seq 5 5 50`; do
    time ./word2vec -train ../Data/mimic_train -output mimic-vectors.bin -cbow 0 -size 200 -window $i -negative 25 -hs 0 -sample 1e-5 -threads 30 -binary 1 -iter 15
    echo $i
    python mimic-accuracy.py
done

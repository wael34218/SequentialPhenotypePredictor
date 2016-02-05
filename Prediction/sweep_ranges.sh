#!/bin/bash
source ~/EnvPythonMimic/bin/activate

for i in `seq 275 25 475`; do
    echo "============ New size $i ============"
    for j in `seq 5 5 50`; do
        echo "size $i - window $j"
        for k in `seq 2 2 10`; do
            echo "decay - $k"
            python3 cbowsim.py --size $i --window $j --decay $k
            python3 collaborative.py --size $i --window $j --decay $k
            python3 skipgram.py --size $i --window $j --decay $k
        done
    done
done

for i in `seq 2 2 10`; do
    echo "============ New ngram $i ============"
    for j in `seq 5 10 45`; do
        echo "ngram $i - skip $j"
        python3 tfidf.py --ngrams $i --skip $j
    done
done

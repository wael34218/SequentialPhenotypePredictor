### Predicting MIMICIII Last Admission Diagnosis


## How to run

This project assumes that you already have access to some MIMICIII database. If not you could follow the following instructions:

https://mimic.physionet.org/tutorials/install_mimic_locally/

You should also install gensim python library: https://radimrehurek.com/gensim/install.html


To run the follow the following instructions:

    cd DataPrep
    psql -U mimic -a -f allevents.sql
    python generate_icd_levels.py
    python generate_sequences.py

    cd ../Prediction
    ./train_model.sh
    python mimic-accuracy.py


## Libraries

This project depends on:

1. word2vec - https://code.google.com/p/word2vec/
2. ICD9 - https://github.com/sirrice/icd9
3. ICD9 - https://github.com/kshedden/icd9
4. gensim - http://radimrehurek.com/gensim/

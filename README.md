### Predicting MIMICIII Last Admission Diagnosis


## How to run

This project assumes that you already have access to some MIMICIII database. If not you could follow the following instructions:

https://mimic.physionet.org/tutorials/install_mimic_locally/

You should also install gensim python library: https://radimrehurek.com/gensim/install.html


To take the code for a spin run the following commands:

    cd DataPrep
    psql -U mimic -a -f allevents.sql
    python generate_icd_levels.py
    python generate_sequences.py

    cd ../Prediction
    python cbowsim.py --window 500 --size 500 --decay 4


## Libraries

This project depends on:

1. word2vec - https://code.google.com/p/word2vec/
2. ICD9 - https://github.com/sirrice/icd9
3. ICD9 - https://github.com/kshedden/icd9
4. gensim - http://radimrehurek.com/gensim/

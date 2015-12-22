time ./../lib/word2vec -train ../Data/mimic_train_0 -output mimic-vectors.bin -cbow 0 -size 200 -window 50 -negative 25 -hs 0 -sample 1e-5 -threads 30 -binary 1 -iter 15

import gensim
import random

model = gensim.models.Word2Vec.load_word2vec_format('mimic-vectors.bin', binary=True)
miss = 0
hit = 0

diags = set()
with open('../Data/mimic_train') as f:
    lines = f.readlines()
    for line in lines:
        events = line[:-1].split(' ')
        diags |= set([x for x in events if x.startswith('d_')])

with open('../Data/mimic_test') as f:
    lines = f.readlines()
    for line in lines:
        feed_index = line[0:line.rfind(" d_")].rfind(",")
        feed_events = line[0:feed_index].replace(",", "").split(" ")

        last_admission = line[feed_index:].replace("\n", "").replace(",", "").split(" ")
        actual = [x for x in last_admission if x.startswith('d_')]
        result = model.most_similar(feed_events, topn=100)
        prediction = []
        prediction += [x for x, d in result if x.startswith('d_')][:4]
        prediction += [x for x in feed_events if x.startswith('d_')]
        # prediction += random.sample(diags, 5)

        if len([x for x in actual if x in prediction]) > 0:
            hit += 1
        else:
            miss += 1


print(hit*1.0/(miss+hit))

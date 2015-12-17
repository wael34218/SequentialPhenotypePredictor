import gensim

hit = 0
miss = 0

diags = set()
print("Generate unique diagnoses list")
for i in range(10):
    with open('../Data/mimic_train_'+str(i)) as f:
        lines = f.readlines()
        for line in lines:
            events = line[:-1].split(' ')
            diags |= set([x for x in events if x.startswith('d_')])

for i in range(10):
    print("Training segment "+str(i))
    with open('../Data/mimic_train_'+str(i)) as f:
        sentences = [s[:-1].split(' ') for s in f.readlines()]
        model = gensim.models.Word2Vec(sentences, sg=1, size=200, window=10,
                                       min_count=5, workers=10)

    seg_hit = 0
    seg_miss = 0

    with open('../Data/mimic_test_'+str(i)) as f:
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
                seg_hit += 1
            else:
                miss += 1
                seg_miss += 1

    print(seg_hit*1.0/(seg_miss + seg_hit))

print("Overall accuracy :")
print(hit*1.0/(miss+hit))

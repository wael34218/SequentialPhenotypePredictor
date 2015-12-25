import gensim
import math
uniq_events = set()
diags = set()
sim_mat = {}

# for size in range(500, 900, 100):
#     for window in range(800, 1300, 100):
window = 500
size = 1000
# decay_coeff = 6

for decay_coeff in range(0, 9):
    hit = 0
    miss = 0
    for i in range(10):
        with open('../Data/mimic_train_'+str(i)) as f:
            sentences = [s[:-1].split(' ') for s in f]
            model = gensim.models.Word2Vec(sentences, sg=0, window=window, size=size, min_count=1, workers=20)

        with open('../Data/mimic_train_'+str(i)) as f:
            lines = f.readlines()
            for line in lines:
                events = line[:-1].split(' ')
                uniq_events |= set(events)
                diags |= set([x for x in events if x.startswith('d_')])

        events_index = sorted(uniq_events)

        for diag in diags:
            words = model.most_similar(diag, topn=200)
            sim_array = [0] * len(uniq_events)
            sim_array[events_index.index(diag)] = 1
            for event, distance in words:
                sim_array[events_index.index(event)] = distance
            sim_mat[diag] = sim_array


        with open('../Data/mimic_test_'+str(i)) as f:
            lines = f.readlines()
            for line in lines:
                feed_index = line[0:line.rfind(" d_")].rfind(",")
                feed_events = line[0:feed_index].replace(",", "").split(" ")
                last_admission = line[feed_index:].replace("\n", "").replace(",", "").split(" ")
                actual = set([x for x in last_admission if x.startswith('d_')])
                test_array = [0] * len(uniq_events)

                te = len(feed_events)
                ev = 1
                for event in feed_events:
                    test_array[events_index.index(event)] += math.exp(decay_coeff*(ev-te)/te)
                    ev += 1

                result = {}
                for diag in sim_mat:
                    result[sum([x*y for x, y in zip(test_array, sim_mat[diag])])] = diag

                distances = sorted(result.keys(), reverse=True)[:5]
                prediction = set([result[x] for x in distances])
                # prediction |= set([x for x in feed_events if x.startswith('d_')])

                if len([x for x in actual if x in prediction]) > 0:
                    hit += 1
                else:
                    miss += 1

    print(decay_coeff, window, size, hit*1.0/(miss+hit))

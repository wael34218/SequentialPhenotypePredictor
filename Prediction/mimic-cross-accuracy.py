import gensim
import csv
import sys

sys.path.insert(0, '../')

import lib.icd9

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

stats = {}
for d in diags:
    stats[d] = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

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

            for act in actual:
                if act in prediction:
                    stats[act]["TP"] += 1
                else:
                    stats[act]["FN"] += 1

            for pred in prediction:
                if pred not in actual:
                    stats[act]["FP"] += 1

            for d in diags:
                if d not in actual and d not in prediction:
                    stats[d]["TN"] += 1

            if len([x for x in actual if x in prediction]) > 0:
                hit += 1
                seg_hit += 1
            else:
                miss += 1
                seg_miss += 1

    print(seg_hit*1.0/(seg_miss + seg_hit))

# Write specificity and sensitivity CSV file
total = hit+miss
print(total)
tree = lib.icd9.ICD9('../lib/icd9/codes.json')
with open('stats.csv', 'w') as csvfile:
    header = ["Stat"]
    desc = ["Description"]
    spec = ["Specificity"]
    sens = ["Sensitivity"]
    for d in diags:
        spec.append(stats[d]["TP"]*1.0 / (stats[d]["TP"] + stats[d]["FN"]))
        # TN = total - stats[d]["TP"] - stats[d]["FN"] - stats[d]["FP"]
        sens.append(stats[d]["TN"]*1.0 / (stats[d]["FP"] + stats[d]["TN"]))
        header.append(d)
        try:
            desc.append(tree.find(d[2:]).description)
        except:
            if d[2:] == "285.9":
                desc.append("Anemia")
            elif d[2:] == "287.5":
                desc.append("Thrombocytopenia")
            elif d[2:] == "285.1":
                desc.append("Acute posthemorrhagic anemia")
            else:
                desc.append('Not Found')

    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerow(desc)
    writer.writerow(spec)
    writer.writerow(sens)

print("Overall accuracy :")
print(hit*1.0/(miss+hit))

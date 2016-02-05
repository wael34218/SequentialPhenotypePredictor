import psycopg2
import math
import datetime
import random
import gensim

coeff = 10.0
delta = 20
uniq_g_feat = set()
uniq_p_feat = ["gender", "age", "white", "asian", "hispanic", "black", "multi", "portuguese",
               "american", "mideast", "hawaiian", "other"]


def set_g_features(t_events):
    feats = {}
    for i, [old_e, old_t] in enumerate(t_events):
        # Delta = 10
        for j in range(i+1, min(i+1+delta, len(t_events))):
            [new_e, new_t] = t_events[j]
            decay = math.exp(((old_t - new_t).days + 1) / coeff)

            similarity = w2v.similarity(new_e, old_e)
            feats[old_e+new_e] = decay * similarity
            uniq_g_feat.add(old_e+new_e)

    return feats


def set_p_features(hadm_id):
    cur.execute("""SELECT dob, admittime, gender, ethnicity from admissions join patients
                on admissions.subject_id = patients.subject_id
                where hadm_id = %(hadm_id)s """ % {'hadm_id': str(hadm_id)})
    subject_info = cur.fetchall()
    feats = {}
    for k in uniq_p_feat:
        feats[k] = 0

    feats["gender"] = int(subject_info[0][2] == "M")
    num_years = (subject_info[0][1] - subject_info[0][0]).days / 365.25
    feats["age"] = num_years

    r = subject_info[0][3]
    if "WHITE" in r:
        feats["white"] = 1
    elif "ASIAN" in r:
        feats["asian"] = 1
    elif "HISPANIC" in r:
        feats["hispanic"] = 1
    elif "BLACK" in r:
        feats["black"] = 1
    elif "MULTI" in r:
        feats["multi"] = 1
    elif "PORTUGUESE" in r:
        feats["portuguese"] = 1
    elif "AMERICAN INDIAN" in r:
        feats["american"] = 1
    elif "MIDDLE EASTERN" in r:
        feats["mideast"] = 1
    elif "HAWAIIAN" in r or "CARIBBEAN" in r:
        feats["hawaiian"] = 1
    else:
        feats["other"] = 1
    return feats


print("Start")
try:
        conn = psycopg2.connect("dbname='mimic' user='mimic' host='localhost' password='mimic'")
except:
        print("I am unable to connect to the database")

cur = conn.cursor()
cur.execute("""set search_path to mimiciii""")
cur.execute("""SELECT subject_id, charttime, event_type, event, icd9_3, hadm_id
            from allevents order by subject_id, charttime, event_type desc, event""")
rows = cur.fetchall()
print("Query executed")

w2v_size = 200
w2v_window = 60

# TODO: allow passed in parameters and redo model for each train set
with open('../Data/w2v/mimic_train_me_0') as f:
    sentences = [s[:-1].replace(",", "").split(' ') for s in f.readlines()]
    w2v = gensim.models.Word2Vec(sentences, sg=0, window=w2v_window, size=w2v_size,
                                 min_count=1, workers=20)

prev_time = None
prev_subject = None
prev_hadm_id = None
diags = set()
total_diags = set()
all_seq = []
e_features = [0] * w2v_size
timed_events = []
temp_e_features = [0] * w2v_size
temp_timed_events = []

for row in rows:
    if row[2] == "diagnosis":
        event = row[2][:1] + "_" + row[4]
    else:
        event = row[2][:1] + "_" + row[3]

    if row[0] is None or row[1] is None or row[5] is None:
        continue

    elif prev_time is None or prev_subject is None:
        pass

    elif (row[0] != prev_subject) or (row[1] > prev_time + datetime.timedelta(365)):
        if len(diags) > 0 and len(timed_events) > 4:
            p_features = set_p_features(row[5])
            g_features = set_g_features(timed_events)
            e_features = [e / len(timed_events) for e in e_features]
            all_seq.append([g_features, p_features, e_features, diags])
        diags = set()
        e_features = [0] * w2v_size
        timed_events = []
        temp_timed_events = []
        temp_e_features = [0] * w2v_size

    elif prev_hadm_id != row[5]:
        timed_events += temp_timed_events
        e_features = [x + y for x, y in zip(e_features, temp_e_features)]

        temp_timed_events = []
        temp_e_features = [0] * w2v_size
        diags = set()

    temp_e_features = [x + y for x, y in zip(w2v[event].tolist(), temp_e_features)]
    temp_timed_events.append([event, row[1]])

    prev_time = row[1]
    prev_subject = row[0]
    prev_hadm_id = row[5]

    if row[2] == "diagnosis":
        diags.add(event)
        total_diags.add(event)

print("Number of total sequences {}".format(len(all_seq)))
print("Data structures created. Now writing files:")
train = {}
test = {}

# To include all diagnoses change it to total_diags
diags = total_diags
for diag in diags:
    for i in range(10):
        train[diag+str(i)] = open('../Data/svd/mimic_train_'+diag+'_'+str(i), 'w')
        test[diag+str(i)] = open('../Data/svd/mimic_test_'+diag+'_'+str(i), 'w')


segment = 0
random.shuffle(all_seq)
total = len(all_seq)
uniq_g_feat = list(uniq_g_feat)

print("Total globals number: {}".format(len(uniq_g_feat)))
print("Total patient number: {}".format(len(uniq_p_feat)))
print("Total event number: {}".format(w2v_size))

for seq_index, seq in enumerate(all_seq):
    if math.floor(seq_index * 10 / total) > segment:
        print("New Segment "+str(segment))
        segment += 1

    [g, p, e, d] = seq
    serial = "{}\t{}\t{}\t".format(len(g), len(p), len(e))
    serial += "\t".join([str(uniq_g_feat.index(k))+":"+str(v) for k, v in g.items()]) + "\t"
    serial += "\t".join([str(uniq_p_feat.index(k))+":"+str(v) for k, v in p.items()]) + "\t"
    serial += "\t".join([str(i)+":"+str(v) for i, v in enumerate(e)])

    for diag in diags:
        test[diag+str(segment)].write(str(int(diag in d))+"\t"+serial+'\n')
    for f in range(10):
        if f != segment:
            for diag in diags:
                train[diag+str(f)].write(str(int(diag in d))+"\t"+serial+'\n')

for diag in diags:
    for i in range(10):
        train[diag+str(i)].close()
        test[diag+str(i)].close()

print("Done")

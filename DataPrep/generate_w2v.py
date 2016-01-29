import psycopg2
import math
import datetime
import random

try:
        conn = psycopg2.connect("dbname='mimic' user='mimic' host='localhost' password='mimic'")
except:
        print("I am unable to connect to the database")

cur = conn.cursor()
cur.execute("""set search_path to mimiciii""")
cur.execute("""SELECT subject_id, charttime, event_type, event, icd9_3, hadm_id
            from allevents order by subject_id, charttime, event_type desc, event""")
rows = cur.fetchall()

index = 0
prev_time = None
prev_subject = None
prev_hadm_id = None
sequence = ""
diags = {}
p_seq = {}
seq_count = 0

for row in rows:
    if prev_time is None or prev_subject is None:
        print("Start")

    elif row[0] is None or row[1] is None:
        continue

    elif (row[0] != prev_subject) or (row[1] > prev_time + datetime.timedelta(365)):
        if len(diags) > 1:
            seq_count += 1
            if prev_subject in p_seq:
                p_seq[prev_subject].append(sequence)
            else:
                p_seq[prev_subject] = [sequence]
        sequence = ""
        diags = {}

    elif prev_hadm_id != row[5]:
        sequence += ", "

    else:
        sequence += " "

    prev_time = row[1]
    prev_subject = row[0]
    prev_hadm_id = row[5]

    if row[2] == "diagnosis":
        diags[row[5]] = 1
        sequence += row[2][:1] + "_" + row[4]
    else:
        sequence += row[2][:1] + "_" + row[3]


train = {}
test = {}

for i in range(10):
    train[i] = open('../Data/w2v/mimic_train_me_'+str(i), 'w')
    test[i] = open('../Data/w2v/mimic_test_me_'+str(i), 'w')


seq_index = 0
segment = 0
keys = list(p_seq.keys())
random.shuffle(keys)

for key in keys:
    if math.floor(seq_index * 10 / seq_count) > segment:
        print("New Segment "+str(segment))
        segment += 1
    for i, seq in enumerate(p_seq[key]):
        test[segment].write(seq+'\n')
        for f in range(10):
            if f != segment:
                train[f].write(seq+'\n')

        seq_index += 1

for i in range(10):
    train[i].close()
    test[i].close()

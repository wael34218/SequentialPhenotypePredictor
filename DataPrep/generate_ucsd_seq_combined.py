import psycopg2
import math
import random
import json
from collections import defaultdict

uniq_p_feat = ["gender", "age", "white", "hispanic", "black", "other", "multi", "pat_id"]


def set_p_features(pat_id):
    cur.execute("""SELECT bday, gender, ethnicity, pat_id from dedemo where pat_id = %(pat_id)s;"""
                , {'pat_id': pat_id})
    subject_info = cur.fetchall()
    feats = {}
    for k in uniq_p_feat:
        feats[k] = 0

    feats["gender"] = int(subject_info[0][1] == "Male")
    feats["age"] = subject_info[0][0]

    r = subject_info[0][2]
    if "Caucasian" in r:
        feats["white"] = 1
    elif "Hispanic/Latino" or "Hispanic" in r:
        feats["hispanic"] = 1
    elif "African American" in r:
        feats["black"] = 1
    elif "Multi-Racuak" in r:
        feats["multi"] = 1
    else:
        feats["other"] = 1

    feats["pat_id"] = subject_info[0][3]
    return feats


print("Start")
try:
        conn = psycopg2.connect("dbname='ucsd' user='mimic' host='localhost' password='mimic'")
except:
        print("I am unable to connect to the database")

cur = conn.cursor()
cur.execute("""SELECT subject_id, charttime, event_type, event, icd9_3, hadm_id
            from allevents order by subject_id, charttime, case event_type when 'symptom' then 1
            when 'labevent' then 2 when 'diagnosis' then 3 when 'prescription' then 4 end, event""")
rows = cur.fetchall()
print("Query executed")

prev_time = None
prev_subject = None
prev_hadm_id = None
diags = set()
total_diags = set()
event_seq = []
temp_event_seq = []
all_seq = []
unique_events = set()
diag_count = defaultdict(lambda: 0)

for row in rows:
    if row[2] == "diagnosis":
        event = row[2][:1] + "_" + row[4]
        if len(event) < 5:
            event = event[:2] + "0" + event[2:]
        diag_count[event] += 1
    else:
        event = row[2][:1] + "_" + row[3]

    if row[0] is None or row[1] is None or row[5] is None:
        continue

    elif prev_time is None or prev_subject is None:
        pass

    elif (row[0] != prev_subject) or (row[1] > prev_time + 365):
        if len(diags) > 0 and len(event_seq) > 4:
            p_features = set_p_features(row[0])
            all_seq.append([p_features, event_seq, temp_event_seq, diags])
        diags = set()
        event_seq = []
        temp_event_seq = []

    elif prev_hadm_id != row[5]:
        event_seq += temp_event_seq
        temp_event_seq = []
        diags = set()

    temp_event_seq.append(event)
    unique_events.add(event)

    prev_time = row[1]
    prev_subject = row[0]
    prev_hadm_id = row[5]

    if row[2] == "diagnosis":
        diags.add(event)
        total_diags.add(event)

uniq = open('../Data/ucsd/uniq', 'w')
uniq.write(' '.join(unique_events) + '\n')
predicted_diags = [y[0] for y in sorted(diag_count.items(), key=lambda x: x[1], reverse=True)[:40]]
uniq.write(' '.join(predicted_diags))
uniq.close()

print("Number of total sequences {}".format(len(all_seq)))
print("Data structures created. Now writing files:")
train = {}
test = {}
valid = {}
trainv = {}


# To include all diagnoses change it to total_diags
for i in range(10):
    train[str(i)] = open('../Data/ucsd/mimic_train_'+str(i), 'w')
    trainv[str(i)] = open('../Data/ucsd/mimic_trainv_'+str(i), 'w')
    test[str(i)] = open('../Data/ucsd/mimic_test_'+str(i), 'w')
    valid[str(i)] = open('../Data/ucsd/mimic_valid_'+str(i), 'w')

segment = 0
random.shuffle(all_seq)
total = len(all_seq)
print(total)


valid_count = 0
for seq_index, seq in enumerate(all_seq):
    if math.floor(seq_index * 10 / total) > segment:
        print("New Segment "+str(segment))
        valid_count = 0
        segment += 1

    [patient, events, final_events, diagnoses] = seq
    serial = (",").join(diagnoses)
    serial += "|" + json.dumps(patient)
    serial += "|" + " ".join(events)
    serial += "|" + " ".join(final_events)

    test[str(segment)].write(serial+'\n')
    for f in range(10):
        if f != segment:
            trainv[str(f)].write(serial+'\n')
            if valid_count < math.floor(total / 10):
                valid[str(f)].write(serial+'\n')
                valid_count += 1
            else:
                train[str(f)].write(serial+'\n')

for i in range(10):
    train[str(i)].close()
    test[str(i)].close()
    valid[str(i)].close()

print("Done")

import psycopg2
import math
import datetime
import random
import json

uniq_p_feat = ["gender", "age", "white", "asian", "hispanic", "black", "multi", "portuguese",
               "american", "mideast", "hawaiian", "other"]


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

prev_time = None
prev_subject = None
prev_hadm_id = None
diags = set()
total_diags = set()
event_seq = []
temp_event_seq = []
all_seq = []

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
        if len(diags) > 0 and len(event_seq) > 4:
            p_features = set_p_features(row[5])
            all_seq.append([p_features, event_seq, temp_event_seq, diags])
        diags = set()
        event_seq = []
        temp_event_seq = []

    elif prev_hadm_id != row[5]:
        event_seq += temp_event_seq
        temp_event_seq = []
        diags = set()

    temp_event_seq.append(event)

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
for i in range(10):
    train[str(i)] = open('../Data/seq_combined/mimic_train_'+str(i), 'w')
    test[str(i)] = open('../Data/seq_combined/mimic_test_'+str(i), 'w')


segment = 0
random.shuffle(all_seq)
total = len(all_seq)

for seq_index, seq in enumerate(all_seq):
    if math.floor(seq_index * 10 / total) > segment:
        print("New Segment "+str(segment))
        segment += 1

    [patient, events, final_events, diagnoses] = seq
    serial = (",").join(diagnoses)
    serial += "|" + json.dumps(patient)
    serial += "|" + " ".join(events)
    serial += "|" + " ".join(final_events)

    test[str(segment)].write(serial+'\n')
    for f in range(10):
        if f != segment:
            train[str(f)].write(serial+'\n')

for i in range(10):
    train[str(i)].close()
    test[str(i)].close()

print("Done")

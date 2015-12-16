import psycopg2
import datetime
import random

try:
        conn = psycopg2.connect("dbname='mimic' user='mimic' host='localhost' password='mimic'")
except:
        print("I am unable to connect to the database")


train = open('../Data/mimic_train', 'w')
test = open('../Data/mimic_test', 'w')

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

for row in rows:
    if prev_time is None or prev_subject is None:
        print("Start")

    elif row[0] is None or row[1] is None:
        continue

    elif (row[0] != prev_subject) or (row[1] > prev_time + datetime.timedelta(365)):
        if len(diags) > 1:
            if row[0] in p_seq:
                p_seq[row[0]].append(sequence)
            else:
                p_seq[row[0]] = [sequence]
        sequence = ""
        diags = {}

    elif prev_hadm_id != row[5]:
        sequence +=", "

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

for key in p_seq.keys():
    index = 0
    for seq in p_seq[key]:
        if random.random() < .1 and index == len(p_seq[key]) - 1:
            test.write(seq+'\n')
        else:
            train.write(seq.replace(",", "")+'\n')

        index += 1

train.close()

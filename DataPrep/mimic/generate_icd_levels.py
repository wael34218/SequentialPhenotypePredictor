import psycopg2
import re

try:
    conn = psycopg2.connect("dbname='mimic' user='mimic' host='localhost' password='mimic'")
except:
    print("I am unable to connect to the database")

cur = conn.cursor()

cur.execute("""set search_path to mimiciii""")
cur.execute("""SELECT id, event from allevents where event_type='diagnosis'""")
rows = cur.fetchall()

index = 0
for row in rows:
    if re.match('^\d{3}', row[1]):
        updateRecordStatus = "update allevents set icd9_3='" + row[1][:3]
    else:
        if row[1][0] != "E":
            print("====>" + row[1])
        updateRecordStatus = "update allevents set icd9_3='" + row[1]
    updateRecordStatus += "' where id=" + str(row[0]) + ";"
    cur.execute(updateRecordStatus)
    conn.commit()

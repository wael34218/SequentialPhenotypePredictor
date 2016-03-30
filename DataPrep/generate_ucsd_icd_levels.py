import os
import sys
lib_path = os.path.abspath(os.path.join('..', 'lib'))
sys.path.append(lib_path)
import psycopg2

try:
        conn = psycopg2.connect("dbname='ucsd' user='mimic' host='localhost' password='mimic'")
except:
        print("I am unable to connect to the database")

cur = conn.cursor()

# cur.execute("""set search_path to mimiciii""")
cur.execute("""SELECT id, event from allevents where event_type='diagnosis'""")
rows = cur.fetchall()

index = 0
for row in rows:
    updateRecordStatus = "update allevents set icd9_3='"+str(int(float(row[1])))+"'"
    updateRecordStatus += " where id=" + str(row[0]) + ";"
    cur.execute(updateRecordStatus)
    conn.commit()

import os
import sys
lib_path = os.path.abspath(os.path.join('..', 'lib'))
sys.path.append(lib_path)

import psycopg2
from icd9 import ICD9
from icd9_converter import short_to_decimal

try:
        conn = psycopg2.connect("dbname='ucsd' user='mimic' host='localhost' password='mimic'")
except:
        print("I am unable to connect to the database")

tree = ICD9('../lib/icd9/codes.json')
cur = conn.cursor()

# cur.execute("""set search_path to mimiciii""")
cur.execute("""SELECT * from allevents where event_type='diagnosis'""")
rows = cur.fetchall()

index = 0
for row in rows:
    updateRecordStatus = "update allevents set icd9_3='"+str(int(float(row[4])))+"'"
    updateRecordStatus += " where id=" + str(row[5]) + ";"
    cur.execute(updateRecordStatus)
    conn.commit()

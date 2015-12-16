import psycopg2
from icd92 import ICD92
import icd9

try:
        conn = psycopg2.connect("dbname='mimic' user='mimic' host='localhost' password='mimic'")
except:
        print("I am unable to connect to the database")

tree = ICD92('codes.json')
cur = conn.cursor()

cur.execute("""set search_path to mimiciii""")
cur.execute("""SELECT * from allevents where event_type='diagnosis'""")
rows = cur.fetchall()

index = 0
for row in rows:
    if row[4][0:1] == "V":
        continue
    newcode = icd9.short_to_decimal(row[4])
    if newcode.find(".") == 2:
        newcode = "0"+newcode
    elif newcode.find(".") == 1:
        newcode = "00" + newcode

    try:
        try:
            if newcode[0:2] == "28":
                parents = ["ROOT", "280-289", newcode[0:3], newcode]
            else:
                parents = [x.code for x in tree.find(newcode).parents]
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            parents = [x.code for x in tree.find(newcode[:-1]).parents]

        updateRecordStatus = "update allevents set icd9_1='"+parents[1]+"', icd9_2='"+parents[2]+"'"
        if len(parents) > 3:
            updateRecordStatus += ", icd9_3='"+parents[3]+"'"

        updateRecordStatus += " where id=" + str(row[5]) + ";"
        cur.execute(updateRecordStatus)
        conn.commit()
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        print("=======> ", row[5], " : ", row[4])

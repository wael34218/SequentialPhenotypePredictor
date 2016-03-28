set search_path to mimiciii;

/* Create allevents table */ 
drop table if exists allevents;
drop sequence if exists allevents_ids;
create table allevents (hadm_id integer, subject_id integer, charttime timestamp, event_type varchar, event varchar);

/* Adding most common abnormal 58 Labevents */
insert into allevents
select hadm_id, subject_id, (array_agg(charttime ORDER BY charttime asc))[1] as charttime, 'labevent' as event_type, itemid as event from labevents
where flag='abnormal' and itemid in
(
  select itemid from
  (
    select hadm_id, itemid from labevents where flag='abnormal' group by hadm_id, itemid
  ) as uniqlab
  group by itemid having count(*) > 5000
)
and subject_id in (select subject_id from admissions group by subject_id having count(*) > 1)
group by itemid, hadm_id, subject_id
;

/* Adding most common 66 Prescriptions */
insert into allevents
select hadm_id, subject_id, starttime as charttime, 'prescription' as event_type, formulary_drug_cd as event from prescriptions
where formulary_drug_cd in
(
  select formulary_drug_cd from
  (
    select hadm_id, formulary_drug_cd from prescriptions group by hadm_id, formulary_drug_cd
  ) as uniqlab
  group by formulary_drug_cd having count(*) > 5000
)
and subject_id in (select subject_id from admissions group by subject_id having count(*) > 1)
;

/* Adding most common 58 Diagnoses */
insert into allevents
select admissions.hadm_id, admissions.subject_id, dischtime as charttime, 'diagnosis' as event_type, icd9_code as event from diagnoses_icd
left join admissions on admissions.hadm_id = diagnoses_icd.hadm_id
where icd9_code in
(
  select icd9_code from
  (
      select hadm_id, icd9_code from diagnoses_icd group by hadm_id, icd9_code
  ) as uniqlab
  group by icd9_code having count(*) > 2000
)
and diagnoses_icd.subject_id in
(select subject_id from admissions group by subject_id having count(*) > 1)
;

/* Delete unimportant diagnoses */
delete from allevents where event_type='diagnosis' and event like 'V%';

/* Create unique ID*/ 
CREATE SEQUENCE allevents_ids;
ALTER TABLE allevents ADD id INT UNIQUE;
UPDATE allevents SET id = NEXTVAL('allevents_ids');

/* Adding icd9 levels */
alter table allevents add column icd9_1 varchar(10);
alter table allevents add column icd9_2 varchar(10);
alter table allevents add column icd9_3 varchar(10);

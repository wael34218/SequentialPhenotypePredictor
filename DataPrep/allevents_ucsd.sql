\c ucsd
drop table if exists allevents;

create table allevents (hadm_id integer, subject_id varchar, charttime integer, event_type varchar, event varchar);

insert into allevents
select encounter_id, pat_id ,(array_agg(result_day ORDER BY result_day asc))[1] as charttime, 'labevent' as event_type, newid as event from delab
where newid in
(
  select newid from
  (
    select encounter_id, newid from delab group by encounter_id, newid
  ) as uniqlab
  group by newid having count(*) > 50
)
and pat_id in (select pat_id from dehosp group by pat_id having count(*) > 1)
group by newid, encounter_id, pat_id
;

/* Adding most common 66 Prescriptions */
insert into allevents
select encounter_id, pat_id, order_day as charttime, 'prescription' as event_type, medicationid as event from demed
where medicationid in
(
  select medicationid from
  (
    select encounter_id, medicationid from demed group by encounter_id, medicationid
  ) as uniqlab
  group by medicationid having count(*) > 50
)
and pat_id in (select pat_id from dehosp group by pat_id having count(*) > 1)
;

/* Adding most common 58 Diagnoses */
insert into allevents
select encounter_id, pat_id, contact_day as charttime, 'diagnosis' as event_type, icd9 as event from dediag
where icd9 in
(
  select icd9 from
  (
      select encounter_id, icd9 from dediag group by encounter_id, icd9
  ) as uniqlab where icd9 < 1000
  group by icd9 having count(*) > 5
)
and pat_id in (select pat_id from dehosp group by pat_id having count(*) > 1)
;

/* Delete unimportant diagnoses */
delete from allevents 
	   where event_type='diagnosis' 
       and  event IS NULL;

delete from allevents 
	   where event_type='prescription' 
       and event IS NULL;

delete from allevents 
	   where event_type='labevent' 
       and event IS NULL ;


/* Create unique ID*/ 
CREATE SEQUENCE allevents_ids;
ALTER TABLE allevents ADD id INT UNIQUE;
UPDATE allevents SET id = NEXTVAL('allevents_ids');

/* Adding icd9 levels */
alter table allevents add column icd9_1 varchar(10);
alter table allevents add column icd9_2 varchar(10);
alter table allevents add column icd9_3 varchar(10);

UPDATE allevents SET event_type='symptom' WHERE event_type='diagnosis' AND (event LIKE '78%' or event LIKE '79%');

-- ------------------------------------------------------------------
-- 1. Patients incldued in  1611233, among them, about 10885 after deletion only has one admission. 
-- 2. 15 different diagnoses 
--    event  
--   --------
--    300
--    401.9
--    250
--    486
--    285.9
--    427.31
--    428
--    311
--    496
--    530.81
--    244.9
--    599
--    564
--    414
--    272.4
--   (15 rows)
-- 3. 48 different labevents
--    event   
--   -----------
--    2109-High
--    2108-High
--    2115-Low
--    2108-Low
--    7104-Low
--    7151-High
--    2117-Low
--    2117-High
--    7100-Low
--    7106-Low
--    2120-Low
--    7109-Low
--    7105-High
--    7111-High
--    2101-High
--    2125-High
--    2104-Low
--    2132-High
--    7106-High
--    7112-High
--    2106-Low
--    2131-High
--    7113-Low
--    7108-High
--    2103-High
--    2356-High
--    1150-High
--    7103-Low
--    7111-Low
--    2109-Low
--    7317-High
--    7114-Low
--    2143-High
--    2121-Low
--    7155-High
--    7114-High
--    2104-High
--    7100-High
--    2118-Low
--    2130-High
--    7107-Low
--    7311-High
--    2107-Low
--    7101-Low
--    2126-High
--    7105-Low
--    7154-High
--    7154-Low
--   (48 rows)
-- 3. 50 different prescriptions
--    event  
--   --------
--    9009
--    27838
--    10011
--    11037
--    10289
--    27692
--    4903
--    62332
--    680
--    18138
--    39058
--    92581
--    10177
--    60540
--    12735
--    101
--    5373
--    451196
--    5172
--    14966
--    450261
--    3757
--    4719
--    2513
--    5002
--    27466
--    2508
--    17405
--    3844
--    4318
--    10467
--    5170
--    7319
--    5940
--    35942
--    11074
--    59059
--    2567
--    800001
--    4452
--    200011
--    16050
--    3037
--    25119
--    17837
--    10814
--    11701
--    1080
--    2365
--    5245
--   (50 rows)
-- 4. 6233 total SEQUENCE
--    total patients in sequences: 6198
--    total patients who have multi-sequences: 35

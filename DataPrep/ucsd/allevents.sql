\c ucsd
drop table if exists allevents;
drop sequence if exists allevents_ids;

create table allevents (hadm_id integer, subject_id varchar, charttime integer, event_type varchar, event varchar, description varchar);

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

/* Adding most prescriptions*/
insert into allevents
select encounter_id, pat_id, order_day as charttime, 'prescription' as event_type, medicationid as event, description from demed
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

/* Adding diagnoses */
insert into allevents
select encounter_id, pat_id, contact_day as charttime, 'diagnosis' as event_type, icd9 as event from dediag
where icd9 in
(
  select icd9 from
  (
      select encounter_id, icd9 from dediag group by encounter_id, icd9
  ) as uniqlab where icd9 !~ '^\d{4}'
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

DELETE FROM allevents where event ~ '^IMO';
DELETE FROM allevents where event ~ '^S';
DELETE FROM allevents where event = 'NULL';
UPDATE allevents SET event_type='symptom' WHERE event_type='diagnosis' AND event ~ '^7[89]\d.*';
UPDATE allevents SET event_type='condition' WHERE event_type='diagnosis' AND event ~ '^V.*';
UPDATE allevents SET event=b.gevent FROM (select (array_agg(event))[1] AS gevent, description FROM allevents WHERE event_type='prescription' GROUP BY description) AS b WHERE allevents.event_type='prescription' AND allevents.description=b.description;

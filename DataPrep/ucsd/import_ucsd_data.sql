DROP DATABASE IF EXISTS ucsd;
CREATE DATABASE ucsd;

\connect ucsd

CREATE TABLE deDemo 
(pat_id varchar, bday int, state varchar, zip int, gender varchar, race varchar, ethnicity varchar, language varchar, maritalStatus varchar, patStatus varchar);

\copy deDemo FROM '../../Data/ucsd_raw/deDemo.csv' WITH (FORMAT CSV, DELIMITER '|', Null 'NaN', HEADER)


CREATE TABLE deDiag 
(pat_id varchar, encounter_id int , ICD9 varchar, Line double precision, primary_dx_YN varchar, contact_day int);

\copy deDiag FROM '../../Data/ucsd_raw/deDiag.csv' WITH (FORMAT CSV, DELIMITER '|', Null 'NaN', QUOTE '''', HEADER)


CREATE TABLE deHosp 
(pat_id varchar, encounter_id int , Admsn_age int, Hosp_Admsn varchar, Admsn_physician varchar, Disch_physician varchar, Disch_Disposition varchar, Dich_destination varchar,	Disch_Dept varchar, insurance varchar,	admin_day int, discharge_day int);

\copy deHosp FROM '../../Data/ucsd_raw/deHosp.csv' WITH (FORMAT CSV, DELIMITER '|', Null 'NaN', QUOTE E'\b', HEADER)


CREATE TABLE deLab
(pat_id varchar, encounter_id int, component_id int, name varchar, ord_num_value double precision , Result_flag varchar, reference_low double precision ,reference_high double precision ,reference_unit  varchar, result_day int);

\copy deLab FROM '../../Data/ucsd_raw/deLab.csv' WITH (FORMAT CSV, DELIMITER '|', HEADER)   


CREATE TABLE deMed
(pat_id varchar, encounter_id int, medicationid BIGINT, description varchar, quantity varchar, OrderStatus varchar, order_day double precision);

\copy deMed FROM '../../Data/ucsd_raw/deMed.csv' WITH (FORMAT CSV, DELIMITER '|', Null 'NULL', quote E'\b', HEADER)

ALTER TABLE deLab ADD newid varchar;
UPDATE deLab SET newid = component_id || '-' || replace(result_flag, ' ', '');

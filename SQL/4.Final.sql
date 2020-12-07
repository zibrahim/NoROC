set search_path to mimicfull; 
drop table if exists NoROC.NoROCSeries cascade; 

set search_path to mimicfull; 
drop table if exists NoROC.TimeSeries cascade; 

create table NoROC.TimeSeries as
	select subject_id, hadm_id, icustay_id, time, value, valuenum, valueuom, vitalid from  NoROC.vitalsfirstday
		where vitalid is not null; 

update NoROC.TimeSeries  
	SET
	valuenum = valuenum*0.45 where vitalid = 'Weightlb';
	

update NoROC.TimeSeries  
	SET
	valuenum = ((valuenum - 32)*5/9) where valuenum > 50 and vitalid = 'Temperature';



drop table if exists NoROC.DemographicsOutcomes;
create table NoROC.DemographicsOutcomes as
	select mimicfull.admissions.subject_id, 
		mimicfull.admissions.hadm_id,  
		case when deathtime is not null then 
				deathtime	
			else null end as deathtime
		, 
		admittime
		,
		DATE_PART('day',dischtime  - admittime) as los
		, gender
		, ROUND((cast(mimicfull.icustays.intime as date) - cast(mimicfull.patients.dob as date))/365.242, 2) AS age
		, dob
			
	from mimicfull.admissions, mimicfull.patients,  mimicfull.elixhauser_quan_score , mimicfull.icustays

	where 
	
	mimicfull.admissions.subject_id = mimicfull.patients.subject_id and mimicfull.admissions.hadm_id in (select hadm_id from NoROC.TimeSeries) and mimicfull.admissions.hadm_id = mimicfull.elixhauser_quan_score.hadm_id and 
	mimicfull.admissions.hadm_id = mimicfull.icustays.hadm_id 
	and mimicfull.elixhauser_quan_score.elixhauser_sid30 > -12000
	and mimicfull.admissions.admission_type !='ELECTIVE' and 
	mimicfull.admissions.admission_type !='NEWBORN';



		
select * from NoROC.demographicsOutcomes where age > 18 and age < 100 and los >= 1 and los < 200
					and (deathtime IS NULL or DATE_PART('day',deathtime  - admittime) > 1); --40458

		
		

set search_path to mimicfull; 
drop table if exists CardiacArrest.CardiacArrestSeries cascade; 

set search_path to mimicfull; 
drop table if exists CardiacArrest.TimeSeries cascade; 

create table CardiacArrest.TimeSeries as
	select subject_id, hadm_id, icustay_id, time, value, valuenum, valueuom, vitalid from  CardiacArrest.vitalsfirstday
		where vitalid is not null and hadm_id in (select hadm_id from mimicfull.diagnoses_icd where icd9_code ='4275'); 

update CardiacArrest.TimeSeries  
	SET
	valuenum = valuenum*0.45 where vitalid = 'Weightlb';
	

update CardiacArrest.TimeSeries  
	SET
	valuenum = ((valuenum - 32)*5/9) where valuenum > 50 and vitalid = 'Temperature';



drop table if exists CardiacArrest.DemographicsOutcomes;
create table CardiacArrest.DemographicsOutcomes as
	select mimicfull.admissions.subject_id, 
		mimicfull.admissions.hadm_id, mimicfull.elixhauser_quan_score.elixhauser_vanwalraven as comorbidity,
		case when deathtime is not null then 
				deathtime	
			else null end as deathtime
		, 
		admittime
		,
		dischtime  - admittime as los
		, gender
		, ROUND((cast(mimicfull.icustays.intime as date) - cast(mimicfull.patients.dob as date))/365.242, 2) AS age
		, dob
			
	from mimicfull.admissions, mimicfull.patients,  mimicfull.elixhauser_quan_score , mimicfull.icustays

	where 
	
	mimicfull.admissions.subject_id = mimicfull.patients.subject_id and mimicfull.admissions.hadm_id in (select hadm_id from CardiacArrest.TimeSeries) and mimicfull.admissions.hadm_id = mimicfull.elixhauser_quan_score.hadm_id and 
	mimicfull.admissions.hadm_id = mimicfull.icustays.hadm_id and
						mimicfull.elixhauser_quan_score.elixhauser_sid30 > -12000;;





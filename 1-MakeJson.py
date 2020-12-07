import pandas as pd
from PythonDataProcessing.Cohort.Cohort import Cohort
from PythonDataProcessing.Cohort.Patient import Patient
from PythonDataProcessing.Processing.Utils import convert_to_datetime, getDay
from PythonDataProcessing.Processing.Clean import clean_cohort
from PythonDataProcessing.Processing.Serialisation import jsonDump

import json

def main():
    configs = json.load(open('PythonDataProcessing/Configuration.json', 'r'))
    data_path = configs['paths']['data_path']

    vitals= pd.read_csv(data_path+"TimeSeries.csv")
    patients= pd.read_csv(data_path+"DemographicsOutcomes.csv")
    patients = clean_cohort(patients, vitals)

    patient_ids = patients['subject_id']

    patients.to_csv("DemographicsOutcomesCleaned.csv")
    cohort = Cohort('icustay_id', "Mortality")

    print("VITALS COLUMNS :", vitals.columns)
    vitals = vitals.loc[vitals['subject_id'].isin(patient_ids)]

    ids = set(vitals.icustay_id)

    i = 0
    for idx in ids :
        print(" LOOPING ", i)
        i = i +1
        subject_id = vitals.loc[vitals['icustay_id'] == idx, 'subject_id']
        print(subject_id.head())
        subject_id = subject_id.iloc[0]
        patient_details = patients.loc[patients['subject_id'] == subject_id]
        print("PATIENT DETAILS: ")
        print(patient_details)
        if not patient_details.empty:
            patient_details = patient_details.iloc[0,:]

            p = Patient(idx, patient_details['los'],  patient_details['gender'], patient_details['age'],
                    patient_details['30DM'], patient_details['admittime'],
                    patient_details['deathtime'])

            cohort.addIndividual(p, idx)
            patientAdmissionDate = patients.loc[patients['subject_id'] == subject_id].loc[:, 'admittime'].values[0]
            patientAdmissionDate = convert_to_datetime(patientAdmissionDate)

            vitals_for_patient = vitals.loc[vitals['icustay_id'] == idx]
            vitals_for_patient.drop(['subject_id', 'hadm_id'], axis=1, inplace=True)
            vitals_for_patient.columns = ['icustay_id', 'time', 'value', 'valuenum', 'valueuom', 'vitalid']

            vitals_for_patient['time'] = pd.to_datetime(vitals_for_patient['time'])
            #vitals_for_patient['time'] = vitals_for_patient['time'].astype(str)

            if vitals_for_patient.shape[0]:
                cohort.addBloodObservations(idx, vitals_for_patient, patientAdmissionDate)

    jsonDump(cohort, data_path+"Cohort.json")

if __name__ == "__main__":
    main()
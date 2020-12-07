import pandas as pd
pd.set_option('display.max_rows', None)

import os
import json

from PythonDataProcessing.Processing.CleanTimeSeries import remove_alpha, remove_nacolumns
def main():

    configs = json.load(open('PythonDataProcessing/Configuration.json', 'r'))
    data_path = configs['paths']['data_path']

    time_series = pd.read_csv(data_path+"OneDaySeriesMortality.csv")

    time_series = remove_alpha(time_series)
    time_series = remove_nacolumns(time_series)
    patient_ids = time_series['PatientID'].unique()

    new_time_series = pd.DataFrame(columns=time_series.columns)

    for p in patient_ids:
        patient_slice = time_series.loc[time_series.PatientID ==p,]
        patient_slice.reset_index()

        cvo2 = patient_slice['CentralvenousO2Saturation']
        creactiveprot = patient_slice['Creactiveprotein']
        if cvo2.isnull().values.all() or creactiveprot.isnull().values.all():
            print(" paitent", p, "has all nan CentralvenousO2Saturation")
            time_series.drop(time_series.loc[time_series.PatientID ==p,:].index, inplace=True)
        #else:
            #new_time_series = new_time_series.append(patient_slice, ignore_index=True)

    new_time_series = time_series.copy()
    int_columns = [ "Day", "Hour", "Age", "Mortality30Days","OrdinalHour"]

    new_time_series[int_columns] = new_time_series[int_columns].astype(int)

    na_columns = set(new_time_series.columns) - set(int_columns)
    na_columns = na_columns - set(['PatientID'])

    float_columns = list(set(new_time_series.columns)  - set(int_columns))
    new_time_series[float_columns] = new_time_series[float_columns].astype(float)

    #print(aggregate_series['PO2/FIO2'].isnull().sum() * 100 /len(aggregate_series['PO2/FIO2']))
    # 1. Identify columns where PO2/FIO2 is null but both FIO2 and PO2 are not null
    #matches = aggregate_series['PO2/FIO2'].isnull() & aggregate_series['FiO2'].notnull() & aggregate_series['PO2'].notnull()
    # 2. Calculate PO2/FIO2 for the columns using the individual PO2 and FIO2 values
    #aggregate_series.loc[matches, 'PO2/FIO2'] = aggregate_series.loc[matches, 'PO2']/aggregate_series.loc[matches, 'FiO2']

    #print(aggregate_series['PO2/FIO2'].isnull().sum() * 100 /len(aggregate_series['PO2/FIO2']))

    #print("dim before remove na ", aggregate_series.shape)
    new_time_series.dropna(axis=1, how='all', inplace=True)

    print(" new time series columns: ", new_time_series.columns)
    log = open("PythonDataProcessing/goat.txt", "w")
    print("test", file=log)
    print(new_time_series.isnull().sum() * 100 /len(new_time_series), file = log)

    new_time_series.to_csv(data_path+"AggrOneDaySeriesSepsis.csv", index=False)

if __name__ == "__main__" :
    main()
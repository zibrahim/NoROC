import json
import pandas as pd

from PythonDataProcessing.Processing.Utils import getDay

configs = json.load(open('Utils/Configuration.json', 'r'))
data_path = configs['paths']['data_path']

#vitals = pd.read_csv(data_path + "TimeSeries.csv")

patients = pd.read_csv(data_path + "DemographicsOutcomes.csv")
print(" All patients: ", patients.shape)
patients = patients[patients.age > 15]
print(" All patients: ", patients.shape)

patients = patients[patients.age < 100]
print(" All patients: ", patients.shape)

patients['days'] = [int(getDay(x)) for x in patients.los]
print(" All patients: ", patients.shape)

patients = patients[patients.days > 0]
print(" All patients: ", patients.shape)

#LOS < 200 DAYS
print(patients.columns)
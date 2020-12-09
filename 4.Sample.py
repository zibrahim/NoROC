import json

from random import sample

from Utils.DataUtils import get_distribution_scalars
import pandas as pd


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

import numpy as np
np.seterr(divide='ignore')

from numpy.random import seed


seed(7)


def main () :
    configs = json.load(open('Utils/Configuration.json', 'r'))
    grouping = configs['data']['grouping']
    dynamic_features = configs['data']['dynamic_columns']

    outcome = configs['data']['classification_outcome']
    timeseries_path = configs['paths']['data_path']

    timeseries = pd.read_csv(timeseries_path+"AggrOneDaySeries.csv")

    y = [int(x) for x in timeseries[outcome].values]
    print("Outcome Distribution: ", get_distribution_scalars(y))
    number_patients = len(list(set(timeseries[grouping])))
    print("number of patients: ", number_patients)
    negative_patients = timeseries.loc[timeseries[outcome] == 0]
    negative_patients = set(negative_patients[grouping])
    number_neg_patients = len(list(negative_patients))
    print("number of negative patients: ", number_neg_patients)

    positive_patients = timeseries.loc[timeseries[outcome] == 1]
    positive_patients = set(positive_patients[grouping])
    num_positive_patients = len(list(positive_patients))
    print("number of positive patients: ", num_positive_patients)

    ##ZI Change this in the real thing
    dataset_size = num_positive_patients*2
    #dataset_size = num_positive_patients/2
    prevalence_rates = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

    for p in prevalence_rates:
        num_positives = int(p*dataset_size)
        positive_samples = sample(population = positive_patients, k = num_positives)
        num_negatives = int((1-p)*dataset_size)
        negative_samples = sample(population = negative_patients, k = num_negatives)
        print(" Prevelance: ", p, "Num Positives: ", len(positive_samples), "Num Negatives: ", len(negative_samples),
              "Total (Sanity Check): ", len(positive_samples)+len(negative_samples))

        positive_df =  timeseries.loc[timeseries[grouping].isin(positive_samples)]
        negative_df = timeseries.loc[timeseries[grouping].isin(negative_samples)]

        rate_df = positive_df.append(negative_df, ignore_index=True)
        rate_df.to_csv(timeseries_path+"Training/"+"TimeSeries"+str(p)+".csv", index=False)

        
if __name__ == '__main__' :
    main()

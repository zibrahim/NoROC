import os
import json

from Utils.DataUtils import flatten, scale, impute, stratified_group_k_fold
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

from pylab import rcParams
import numpy as np
np.seterr(divide='ignore')

from numpy.random import seed

import os.path

seed(7)

rcParams['figure.figsize'] = 8, 6
LABELS = ["0", "1"]



def main () :
    configs = json.load(open('Utils/Configuration.json', 'r'))
    data_path = configs['paths']['data_path']
    timeseries_path = data_path+"Training/"
    grouping = configs['data']['grouping']
    dynamic_features = configs['data']['dynamic_columns']

    outcome = configs['data']['classification_outcome']
    batch_size = configs['data']['batch_size']

    prevalence_rates = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

    for p in prevalence_rates:
        timeseries = pd.read_csv(timeseries_path+"TimeSeries"+str(p)+".csv")

        y = np.array([int(x) for x in timeseries[outcome].values])
        groups = np.array(timeseries[grouping])

        timeseries = scale(timeseries, dynamic_features)
        timeseries = impute(timeseries,dynamic_features)
        X = timeseries[dynamic_features]
        X.reset_index()



        for ffold_ind, (training_ind, testing_ind) in enumerate(
                stratified_group_k_fold(y, groups, k=10)) :  # CROSS-VALIDATION
            training_groups, testing_groups = groups[training_ind], groups[testing_ind]
            y_train, y_test = y[training_ind], y[testing_ind]
            X_train, X_test = X.iloc[training_ind], X.iloc[testing_ind]
            assert len(set(training_groups) & set(testing_groups)) == 0

            print("Training length: ", X_train.shape, len(y_train) )
            print(" Testing length: ", X_test.shape, len(y_test))

if __name__ == '__main__' :
    main()

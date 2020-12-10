import json
from sktime.transformers.series_as_features.compose import ColumnConcatenator
from sklearn.pipeline import Pipeline

from Prediction.Metrics import performance_metrics
from Utils.DataUtils import scale, impute, stratified_group_k_fold, get_distribution
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

from pylab import rcParams
import numpy as np
np.seterr(divide='ignore')

from numpy.random import seed
from sktime.classification.compose import TimeSeriesForestClassifier

seed(7)

rcParams['figure.figsize'] = 8, 6
LABELS = ["0", "1"]



def main () :
    configs = json.load(open('Utils/Configuration.json', 'r'))
    data_path = configs['paths']['data_path']
    timeseries_path = data_path+"Training/"
    output_path = data_path+"Output/"

    grouping = configs['data']['grouping']
    dynamic_features = configs['data']['dynamic_columns']

    outcome = configs['data']['classification_outcome']
    batch_size = configs['data']['batch_size']

    prevalence_rates = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

    for p in prevalence_rates:
        timeseries = pd.read_csv(timeseries_path+"TimeSeries"+str(p)+".csv")
        y = np.array([int(x) for x in timeseries[outcome].values])
        groups = np.array(timeseries[grouping])

        #timeseries = scale(timeseries, dynamic_features)
        #timeseries = impute(timeseries,dynamic_features)
        X = timeseries[dynamic_features]
        X.reset_index()
        print(" X shape: ", X.shape)
        ##Create Classifiers:

        steps = [
            ("concatenate", ColumnConcatenator()),
            ("classify", TimeSeriesForestClassifier(n_estimators=100)),
        ]
        clf = Pipeline(steps)
        clf_name = "TSForest"

        perf_df = pd.DataFrame()

        for ffold_ind, (training_ind, testing_ind) in enumerate(
                stratified_group_k_fold(y, groups, k=5)) :  # CROSS-VALIDATION
            training_y_ind = [x for x in training_ind if x%48 ==0]
            testing_y_ind = [x for x in testing_ind if x%48 ==0]


            training_groups, testing_groups = groups[training_ind], groups[testing_ind]
            y_train, y_test = y[training_y_ind], y[testing_y_ind]
            print(" ytrain distribution: ", get_distribution(y_train))
            print("ytest distribution: ", get_distribution(y_test))

            X_train, X_test = (X.iloc[training_ind]).to_numpy(), (X.iloc[testing_ind]).to_numpy()
            X_train = X_train.reshape(-1,48,31)
            X_test = X_test.reshape(-1,48,31)

            assert len(set(training_groups) & set(testing_groups)) == 0

            ###########################TimeSeriesForest
            clf.fit(X_train, y_train)
            clf_preds = clf.predict(X_test)

            perf_dict = performance_metrics(y_test, clf_preds)
            perf_df = perf_df.append(perf_dict, ignore_index=True)

        perf_df.to_csv(output_path + "Out"+clf_name+str(p)+".csv", index=False)


if __name__ == '__main__' :
    main()

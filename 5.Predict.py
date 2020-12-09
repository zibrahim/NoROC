import os
import json

from Utils.DataUtils import scale, impute, stratified_group_k_fold
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

from pylab import rcParams
import numpy as np
np.seterr(divide='ignore')

from numpy.random import seed

import os.path

from sklearn.metrics import precision_recall_curve

from sktime.classification.compose import ColumnEnsembleClassifier,TimeSeriesForestClassifier

from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.classification.shapelet_based import MrSEQLClassifier

from sklearn.metrics import roc_auc_score, auc

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
        print(" X shape: ", X.shape)


        ##Create Classifiers:

        clfTSF = ColumnEnsembleClassifier(
            estimators=[
                ("TSF0", TimeSeriesForestClassifier(n_estimators=100), [0]),
                ("TSF1", TimeSeriesForestClassifier(n_estimators=100), [1]),
                ("TSF2", TimeSeriesForestClassifier(n_estimators=100), [2]),
                ("TSF3", TimeSeriesForestClassifier(n_estimators=100), [3]),
                ("TSF4", TimeSeriesForestClassifier(n_estimators=100), [4]),
                ("TSF5", TimeSeriesForestClassifier(n_estimators=100), [5]),
                ("TSF6", TimeSeriesForestClassifier(n_estimators=100), [6]),
                ("TSF7", TimeSeriesForestClassifier(n_estimators=100), [7]),
                ("TSF8", TimeSeriesForestClassifier(n_estimators=100), [8]),
                ("TSF9", TimeSeriesForestClassifier(n_estimators=100), [9]),
                ("TSF10", TimeSeriesForestClassifier(n_estimators=100), [10]),
                ("TSF11", TimeSeriesForestClassifier(n_estimators=100), [11]),
                ("TSF12", TimeSeriesForestClassifier(n_estimators=100), [12]),
                ("TSF13", TimeSeriesForestClassifier(n_estimators=100), [13]),
                ("TSF14", TimeSeriesForestClassifier(n_estimators=100), [14]),
                ("TSF15", TimeSeriesForestClassifier(n_estimators=100), [15]),
                ("TSF16", TimeSeriesForestClassifier(n_estimators=100), [16]),
                ("TSF17", TimeSeriesForestClassifier(n_estimators=100), [17]),
                ("TSF18", TimeSeriesForestClassifier(n_estimators=100), [18]),
                ("TSF19", TimeSeriesForestClassifier(n_estimators=100), [19]),
                ("TSF20", TimeSeriesForestClassifier(n_estimators=100), [20]),
                ("TSF21", TimeSeriesForestClassifier(n_estimators=100), [21]),
                ("TSF22", TimeSeriesForestClassifier(n_estimators=100), [22]),
                ("TSF23", TimeSeriesForestClassifier(n_estimators=100), [23]),
                ("TSF24", TimeSeriesForestClassifier(n_estimators=100), [24]),
                ("TSF25", TimeSeriesForestClassifier(n_estimators=100), [25]),
                ("TSF26", TimeSeriesForestClassifier(n_estimators=100), [26]),
                ("TSF27", TimeSeriesForestClassifier(n_estimators=100), [27]),
                ("TSF28", TimeSeriesForestClassifier(n_estimators=100), [28]),
                ("TSF29", TimeSeriesForestClassifier(n_estimators=100), [29]),
                ("TSF30", TimeSeriesForestClassifier(n_estimators=100), [30])
            ]
        )

        clfMrSEQL = MrSEQLClassifier()
        clfBOSS = ColumnEnsembleClassifier(
            estimators=[
                ("BOSSEnsemble0", BOSSEnsemble(max_ensemble_size=None), [0]),
                ("BOSSEnsemble1", BOSSEnsemble(max_ensemble_size=None), [1]),
                ("BOSSEnsemble2", BOSSEnsemble(max_ensemble_size=None), [2]),
                ("BOSSEnsemble3", BOSSEnsemble(max_ensemble_size=None), [3]),
                ("BOSSEnsemble4", BOSSEnsemble(max_ensemble_size=None), [4]),
                ("BOSSEnsemble5", BOSSEnsemble(max_ensemble_size=None), [5]),
                ("BOSSEnsemble6", BOSSEnsemble(max_ensemble_size=None), [6]),
                ("BOSSEnsemble7", BOSSEnsemble(max_ensemble_size=None), [7]),
                ("BOSSEnsemble8", BOSSEnsemble(max_ensemble_size=None), [8]),
                ("BOSSEnsemble9", BOSSEnsemble(max_ensemble_size=None), [9]),
                ("BOSSEnsemble10", BOSSEnsemble(max_ensemble_size=None), [10]),
                ("BOSSEnsemble11", BOSSEnsemble(max_ensemble_size=None), [11]),
                ("BOSSEnsemble12", BOSSEnsemble(max_ensemble_size=None), [12]),
                ("BOSSEnsemble13", BOSSEnsemble(max_ensemble_size=None), [13]),
                ("BOSSEnsemble14", BOSSEnsemble(max_ensemble_size=None), [14]),
                ("BOSSEnsemble15", BOSSEnsemble(max_ensemble_size=None), [15]),
                ("BOSSEnsemble16", BOSSEnsemble(max_ensemble_size=None), [16]),
                ("BOSSEnsemble17", BOSSEnsemble(max_ensemble_size=None), [17]),
                ("BOSSEnsemble18", BOSSEnsemble(max_ensemble_size=None), [18]),
                ("BOSSEnsemble19", BOSSEnsemble(max_ensemble_size=None), [19]),
                ("BOSSEnsemble20", BOSSEnsemble(max_ensemble_size=None), [20]),
                ("BOSSEnsemble21", BOSSEnsemble(max_ensemble_size=None), [21]),
                ("BOSSEnsemble22", BOSSEnsemble(max_ensemble_size=None), [22]),
                ("BOSSEnsemble23", BOSSEnsemble(max_ensemble_size=None), [23]),
                ("BOSSEnsemble24", BOSSEnsemble(max_ensemble_size=None), [24]),
                ("BOSSEnsemble25", BOSSEnsemble(max_ensemble_size=None), [25]),
                ("BOSSEnsemble26", BOSSEnsemble(max_ensemble_size=None), [26]),
                ("BOSSEnsemble27", BOSSEnsemble(max_ensemble_size=None), [27]),
                ("BOSSEnsemble28", BOSSEnsemble(max_ensemble_size=None), [28]),
                ("BOSSEnsemble29", BOSSEnsemble(max_ensemble_size=None), [29]),
                ("BOSSEnsemble30", BOSSEnsemble(max_ensemble_size=None), [30])
            ]
        )

        for ffold_ind, (training_ind, testing_ind) in enumerate(
                stratified_group_k_fold(y, groups, k=5)) :  # CROSS-VALIDATION
            training_y_ind = [x for x in training_ind if x%48 ==0]
            testing_y_ind = [x for x in testing_ind if x%48 ==0]

            training_groups, testing_groups = groups[training_ind], groups[testing_ind]
            y_train, y_test = y[training_y_ind], y[testing_y_ind]

            X_train, X_test = (X.iloc[training_ind]).to_numpy(), (X.iloc[testing_ind]).to_numpy()
            X_train = X_train.reshape(-1,48,31)
            X_test = X_test.reshape(-1,48,31)

            assert len(set(training_groups) & set(testing_groups)) == 0

            ###########################TimeSeriesForest
            clfTSF.fit(X_train, y_train)
            clf_preds = clfTSF.predict(X_test)
            clf_roc_auc = roc_auc_score(y_test, clf_preds)
            precision_rt, recall_rt, threshold_rt = precision_recall_curve(y_test,
                                                                           clf_preds)
            clf_pr_auc = auc(precision_rt, recall_rt)
            print("TSF ROC-AUC: " + str(clf_roc_auc), "PR-AUC:", clf_pr_auc)
            clf_pr_auc = auc(precision_rt, recall_rt)
            print("TSF ROC-AUC: " + str(clf_roc_auc), "PR-AUC:", clf_pr_auc)
            ###########################
            clfMrSEQL.fit(X_train, y_train)
            clf2_preds = clfMrSEQL.predict(X_test)
            clf2_roc_auc = roc_auc_score(y_test, clf2_preds)
            precision_rt2, recall_rt2, threshold_rt2 = precision_recall_curve(y_test,
                                                                           clf2_preds)
            clf2_pr_auc = auc(precision_rt2, recall_rt2)
            print("TSF2 ROC-AUC: " + str(clf2_roc_auc), "PR-AUC:", clf2_pr_auc)
            ###########################
            clfBOSS.fit(X_train, y_train)
            clf3_preds = clfBOSS.predict(X_test)
            clf3_roc_auc = roc_auc_score(y_test, clf3_preds)
            precision_rt3, recall_rt3, threshold_rt3 = precision_recall_curve(y_test,
                                                                           clf3_preds)
            clf3_pr_auc = auc(precision_rt3, recall_rt3)
            print("TSF3 ROC-AUC: " + str(clf3_roc_auc), "PR-AUC:", clf3_pr_auc)

if __name__ == '__main__' :
    main()

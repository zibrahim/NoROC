import os
import json

from sklearn.metrics import auc

from Prediction.LSTMAutoEncoder.LSTMAutoEncoder import LSTMAutoEncoder
from Prediction.XGBoost.XGBoost import XGBoostClassifier

from Prediction.LSTMAutoEncoder.Utils import process_data, lstm_flatten
from Prediction.ProcessResults.ClassificationReport import ClassificationReport
from Utils.DataUtils import flatten, scale, impute

from sktime.classification.frequency_based import RandomIntervalSpectralForest
from sktime.transformers.series_as_features.compose import ColumnConcatenator
from sklearn.pipeline import Pipeline

from Prediction.Metrics import performance_metrics
from Utils.DataUtils import scale, impute, stratified_group_k_fold
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

from pylab import rcParams
import numpy as np
np.seterr(divide='ignore')

from numpy.random import seed
from Utils.DataUtils import class_weights, class_counts, get_train_test_split, generate_aggregates

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
    #static_features = configs['data']['static_features']
    outcome = configs['data']['classification_outcome']
    batch_size = configs['data']['batch_size']
    epochs = configs['training']['epochs']

    autoencoder_models_path = configs['paths']['autoencoder_models_path']

    prevalence_rates = [0.1, 0.2, 0.3, 0.4, 0.5]

    classification_report = ClassificationReport()

    #save lstm performance for comparison with final outcome
    lstm_praucs = []

    for p in prevalence_rates:
        timeseries = pd.read_csv(timeseries_path+"PneumoniaTimeSeries"+str(p)+".csv")
        y = np.array([int(x) for x in timeseries[outcome].values])
        groups = np.array(timeseries[grouping])

        X = timeseries[dynamic_features]
        X.reset_index()
        print(" X shape: ", X.shape)
        ##Create Classifiers:

        fold_ind, train_ind, test_ind = get_train_test_split(timeseries[outcome].astype(int),
                                                             timeseries[grouping])

        ##Load LSTM models if they exist, otherwise train new models and save them
        autoencoder_filename = autoencoder_models_path + configs['model']['name'] + str(p) +'.h5'
        X_train, X_train_y0, X_valid_y0, X_valid, y_valid, X_test, y_test, timesteps, \
        n_features = \
            process_data(timeseries, outcome, grouping, batch_size,train_ind, test_ind)
        if os.path.isfile(autoencoder_filename) :
                print(" Autoencoder trained model exists for ", p, "file:", autoencoder_filename)
                autoencoder = LSTMAutoEncoder(configs['model']['name'] + str(p), outcome, str(p),
                                              timesteps, n_features, saved_model=autoencoder_filename)
                autoencoder.summary()


        else :
                print("Autencoder trained model does not exist for ", p, "file:", autoencoder_filename)
                autoencoder = LSTMAutoEncoder(configs['model']['name']  + str(p) , outcome, str(p), timesteps, n_features)
                autoencoder.summary()

                autoencoder.fit(X_train_y0, X_train_y0, epochs, batch_size, X_valid_y0, X_valid_y0, 2, str(p))
                autoencoder.plot_history()

        train_x_predictions = autoencoder.predict(X_train)
        mse_train = np.mean(np.power(lstm_flatten(X_train) - lstm_flatten(train_x_predictions), 2), axis=1)

        test_x_predictions = autoencoder.predict(X_test)
        mse_test = np.mean(np.power(lstm_flatten(X_test) - lstm_flatten(test_x_predictions), 2), axis=1)

        test_error_df = pd.DataFrame({'Reconstruction_error' : mse_test,
                                          'True_class' : y_test.tolist()})

        pred_y, best_threshold, precision_rt, recall_rt = \
                autoencoder.predict_binary(test_error_df.True_class, test_error_df.Reconstruction_error)

        prediction_probabilities = [np.sqrt(x) for x in test_error_df.Reconstruction_error]
        autoencoder.output_performance(test_error_df.True_class, pred_y, prediction_probabilities)

        autoencoder.plot_reconstruction_error(test_error_df, best_threshold)
        autoencoder.plot_roc(test_error_df)
        autoencoder.plot_pr(precision_rt, recall_rt)
        lstm_prauc = auc(recall_rt, precision_rt)
        lstm_praucs.append(lstm_prauc)

        perf_df = pd.DataFrame()
        perf_dict = performance_metrics(test_error_df.True_class, pred_y,prediction_probabilities)
        perf_df = perf_df.append(perf_dict, ignore_index=True)

        perf_df.to_csv(output_path + "Out"+str(p)+".csv", index=False)


if __name__ == '__main__' :
    main()

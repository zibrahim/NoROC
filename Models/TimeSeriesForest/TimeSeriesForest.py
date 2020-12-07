from pyts.classification import TimeSeriesForest
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from Models.Metrics import performance_metrics
from Models.Utils import get_distribution
import json
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

class TSForest():
    def __init__(self, X, y,outcome, grouping, saved_model = None):

        self.predicted_probabilities = pd.DataFrame()
        self.X = X
        self.y = y.astype(int)
        self.outcome = outcome
        self.grouping = grouping
        configs = json.load(open('Configuration.json', 'r'))
        self.output_path = configs['paths']['tsforest_output_path']

        class_distributions = [get_distribution(y.astype(int))]
        class_weights = {0:class_distributions[0][0], 1:class_distributions[0][1]}
        #class_weights = class_distributions[0][0] / class_distributions[0][1]

        self.model = TimeSeriesForest(random_state=43, class_weight = class_weights)


    def fit(self, label, sample_weights):
        x_columns = ((self.X.columns).tolist())
        X = self.X[x_columns]
        X.reset_index()
        y = self.y
        y.reset_index()

        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        scores = cross_validate(self.model.fit(X,y),
                                X,y, scoring = ['f1_macro', 'precision_macro',
                               'recall_macro'], cv = cv
                )


        print(label+'Mean F1 Macro:', np.mean(scores['test_f1_macro']), 'Mean Precision Macro: ',
              np.mean(scores['test_precision_macro']), 'mean Recall Macro' ,
              np.mean(scores['test_recall_macro']))

        #self.model.fit(X,y,sample_weight=sample_weights)
        #return predicted_Y, predicted_thredholds, predicted_IDs, self.model.feature_importances_


    def predict( self, holdout_X, holdout_y):

        x_columns = ((holdout_X.columns).tolist())
        #x_columns.remove(self.grouping)

        holdout_X = holdout_X[x_columns]
        holdout_X.reset_index()

        yhat = (self.model).predict_proba(holdout_X)[:, 1]
        precision_rt, recall_rt, thresholds = precision_recall_curve(holdout_y, yhat)
        fscore = (2 * precision_rt * recall_rt) / (precision_rt + recall_rt)

        ix = np.argmax(fscore)
        best_threshold = thresholds[ix]
        y_pred_binary = (yhat > thresholds[ix]).astype('int32')

        return y_pred_binary, best_threshold, precision_rt, recall_rt, yhat


    def plot_pr( self, precision, recall, label):
        pr_auc =  auc(recall, precision)
        plt.figure(figsize=(10, 10))
        plt.plot(recall, precision, linewidth=5, label='PR-AUC = %0.3f' % pr_auc)
        plt.plot([0, 1], [1, 0], linewidth=5)

        plt.xlim([-0.01, 1])
        plt.ylim([0, 1.01])
        plt.legend(loc='lower right')
        plt.title(self.outcome+' Precision Recall Curive-'+label)
        plt.ylabel('Precision')
        plt.xlabel('Recall')

        plt.savefig(self.output_path+self.outcome+label+"precision_recall_auc.pdf", bbox_inches='tight')

    def plot_feature_importance( self ,colnames):
        plt.figure(figsize=(10, 10))
        plt.bar(range(len((self.model).feature_importances_)), (self.model).feature_importances_)
        plt.xticks(range(len((self.model).feature_importances_)), colnames, rotation='vertical')
        plt.savefig(self.output_path + self.outcome  + "xgbprecision_recall_auc.pdf", bbox_inches='tight')

    def output_performance ( self, true_class, pred_y ) :
        perf_df = pd.DataFrame()
        perf_dict = performance_metrics(true_class, pred_y)
        perf_df = perf_df.append(perf_dict, ignore_index=True)
        perf_df.to_csv(self.output_path + "xgboostperformancemetrics" + self.outcome + ".csv", index=False)


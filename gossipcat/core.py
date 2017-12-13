#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import itertools
from lightgbm import LGBMClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split
from simulated_annealing.optimize import SimulatedAnneal
import matplotlib.pyplot as plt


def glimpse(data, target):
    """
    This function prints a general infomation of the dataset and plot the
    distribution of target value.
    """
    print('\nInformation:')
    print(data.info())
    print('\nHead:')
    print(data.head())
    print('\nShape:')
    print(data.shape)
    print('\nTarget Rate:')
    print(dataset[target].values.sum() / dataset.shape[0])
    print('\nCorrelated Predictors(0.9999):')
    print(corr_pairs(dataset[predictors], gamma=0.9999))
    print('\nCorrelated Predictors(0.9):')
    print(corr_pairs(dataset[predictors], gamma=0.9))
    print('\nCorrelated Predictors(0.85):')
    print(corr_pairs(dataset[predictors], gamma=0.85))
    print('\nTarget Distribution:')
    data[target].plot.hist()

    return None


def features_dup(df, n_head=5000, print_dup=False):
    """
    This function checks first n_head rows and obtains duplicated features.
    """
    if dataset.head(n_head).T.duplicated().any():
        dup_list = np.where(df.head(n_head).T.duplicated())[0].tolist()
        dup_features = df.columns[dup_list]
        if print_dup:
            print(dup_features)
    return dup_features.tolist()


def features_clf(df, features):
    """
    This function divides features into sublists according to their data type.
    """
    dtypes = df[features].dtypes.apply(lambda x: x.name).to_dict()
    int_features, float_features, object_features = [], [], []
    for col, dtype in dtypes.items():
        if dtype == 'int64':
            int_features.append(col)
        elif dtype == 'float64':
            float_features.append(col)
        elif dtype == 'object':
            object_features.append(col)
    return int_features, float_features, object_features


def corr_pairs(df, gamma=0.99999):
    """
    This function computes correlated feature pairs with correlated coefficient 
    larger than gamma.
    """
    corr_matrix = df.corr().abs()
    os = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
          .stack()
          .sort_values(ascending=False))
    return os[os > gamma].index.values.tolist()


def features_new(df, ls, target, auc_score=0.75, silent=False):
    """
    This function builds new features based on correlated feature pairs
    if the new feature has an auc greater than auc_score.
    """
    new = pd.DataFrame()
    for index, value in enumerate(ls):
        temp = df[ls[index][0]] - df[ls[index][1]]
        if len(temp.unique()) > 1:
            temp = pd.DataFrame(temp.fillna(temp.median()))
            lr = LR()
            lr.fit(temp, df[target])
            prob = lr.predict_proba(temp)[:, 1]
            auc = metrics.roc_auc_score(df[target], prob)
            if auc > auc_score:
                if silent:
                    print('-'.join(value), ' AUC (train): ', auc)
                new['-'.join(value)] = df[ls[index][0]] - df[ls[index][1]]
    return new


def simAnneal(Train, predictors, target, param, results=True):
    """
    This function uses the simulated annealing to find the optimal hyper 
    parameters and return an optimized classifier.
    """
    print('\nsimulating...')
    gbm = LGBMClassifier(
        learning_rate=0.01, n_estimators=5000, objective='binary', metric='auc',
        save_binary=True, is_unbalance=True, random_state=2017
    )
    sa = SimulatedAnneal(gbm, param, T=10.0, T_min=0.001, alpha=0.75,
                         verbose=True, max_iter=0.25, n_trans=5, max_runtime=300,
                         cv=3, scoring='roc_auc', refit=True)
    sa.fit(Train[predictors].as_matrix(), Train[target].as_matrix())
    if results:
        print('\n best score: ', sa.best_score_,
              '\n best parameters: ', sa.best_params_)
    optimized_clf = sa.best_estimator_

    return optimized_clf


def report(clf, Train, Test, predictors, target):
    """
    This function prints model report with a classifier on test dataset.
    """
    print('\npredicting...')
    dtrain_predictions = clf.predict(Train[predictors])
    dtest_predictions = clf.predict(Test[predictors])
    dtrain_predprob = clf.predict_proba(Train[predictors])[:, 1]
    dtest_predprob = clf.predict_proba(Test[predictors])[:, 1]

    print("\nModel Report")
    print("Accuracy : %f" % metrics.accuracy_score(
        Train[target], dtrain_predictions))
    print("AUC Score (Train): %f" %
          metrics.roc_auc_score(Train[target], dtrain_predprob))
    print('AUC Score (Test): %f' %
          metrics.roc_auc_score(Test[target], dtest_predprob))
    print(classification_report(Test[target], dtest_predictions))

    return None


def plot_confusion_matrix(cm, classes=[0, 1],
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        1  # print('Confusion matrix, without normalization')
    # print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return None


def report_CM(clf, Test, predictors, target):
    """
    This function prints the recall rate of the classifier on test data and 
    plots out confusion matrix.
    """
    print('\npredicting...')
    dtest_predictions = clf.predict(Test[predictors])
    print("\nModel Report")
    cnf_matrix = confusion_matrix(Test[target], dtest_predictions)
    np.set_printoptions(precision=2)
    print("Recall metric in the testing dataset: ", cnf_matrix[
        1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))
    plt.figure()
    plot_confusion_matrix(cnf_matrix)

    return None


def report_PR(clf, Test, predictors, target):
    """
    This function plots precision-recall curve and gives average precision.
    """
    print('\npredicting...')
    dtest_predprob = clf.predict_proba(Test[predictors])[:, 1]
    precision, recall, _ = precision_recall_curve(Test[target], dtest_predprob)
    average_precision = average_precision_score(Test[target], dtest_predprob)
    print('\nModel Report')
    print('Average Precision: {0:0.4f}'.format(average_precision))
    plt.figure(figsize=(8, 7))

    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.5, color='red')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

    return None


def main():
    pass

if __name__ == '__main__':
    main()

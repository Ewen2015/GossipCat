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


param_1 = {
    'max_depth': [i for i in range(1, 10, 1)],
    'subsample': [i / 10.0 for i in range(1, 10, 1)],
    'colsample_bytree': [i / 10.0 for i in range(1, 10, 1)],
}


def glimpse(data, target, predictors):
    """A glimpse at the dataset.

    Prints a general infomation of the dataset and plot the distribution 
    of target value.

    Args:
        data: A dataset you wanna glimpse.
        target: The target variable in your dataset; limited to binary.
        predictors: The predictors of your dataset.
    """
    print('\nInformation:')
    print(data.info())
    print('\nHead:')
    print(data.head())
    print('\nShape:')
    print(data.shape)
    print('\nTarget Rate:')
    print(data[target].values.sum() / data.shape[0])
    print('\nCorrelated Predictors(0.9999):')
    print(corr_pairs(data[predictors], gamma=0.9999))
    print('\nCorrelated Predictors(0.9):')
    print(corr_pairs(data[predictors], gamma=0.9))
    print('\nCorrelated Predictors(0.85):')
    print(corr_pairs(data[predictors], gamma=0.85))
    print('\nTarget Distribution:')
    data[target].plot.hist()

    return None


def features_dup(data, n_head=5000, print_dup=False):
    """ Obtain duplicated features.

    Checks first n_head rows and obtains duplicated features.

    Args:
        data: A dataset you wanna check and return duplicated columns names.
        n_head: First n_head rows to be checked; default 5000.
        print_dup: Whether print duplicated columns names; default with False.

    Returns:
        A list of the names of duplicatec columns.
    """
    dup_features = []
    if data.head(n_head).T.duplicated().any():
        dup_list = np.where(data.head(n_head).T.duplicated())[0].tolist()
        dup_features = data.columns[dup_list]
        if print_dup:
            print(dup_features)

    return dup_features.tolist()


def features_clf(data, features):
    """ Feature classification.

    Divides features into sublists according to their data type.

    Args:
        data: A dataset which you wanna classify features into subsets 
            according to the data type.
        features: A list of column names.

    Returns:
        int_features: A list of column names of int features.
        float_features: A list of column names of float features.
        object_features: A list of column names of object features.
    """
    dtypes = data[features].dtypes.apply(lambda x: x.name).to_dict()
    int_features, float_features, object_features = [], [], []

    for col, dtype in dtypes.items():
        if dtype == 'int64':
            int_features.append(col)
        elif dtype == 'float64':
            float_features.append(col)
        elif dtype == 'object':
            object_features.append(col)

    return int_features, float_features, object_features


def corr_pairs(data, gamma=0.9):
    """ Detect corralted features.

    Computes correlated feature pairs with correlated coefficient larger than gamma.

    Args:
        data: A dataset which you wanna detect corralted features from.
        gamma: The correlated coefficiency; default at 0.9.

    Returns:
        pairs: A list of correlated features.
    """
    pairs = []
    corr_matrix = data.corr().abs()
    os = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
          .stack()
          .sort_values(ascending=False))
    pairs = os[os > gamma].index.values.tolist()
    return pairs


def features_new(data, corr_list, target, auc_score=0.75, silent=False):
    """ Build new features from correlated features.

    Builds new features based on correlated feature pairs if the new feature 
    has an auc greater than auc_score.

    Args:
        data: A dataset which you wanna generate new features from.
        corr_list: The correlated list to generate new features from.
        target: The target variable in your dataset; limited to binary.
        auc_score: The auc to decide whether generate features, default at 0.75.
        silent: Whether print the new features' names out; default with False.

    Returns:
        new: A dataset conatianing new features.
    """
    new = pd.DataFrame()

    for index, value in enumerate(corr_list):
        temp = data[corr_list[index][0]] - data[corr_list[index][1]]
        if len(temp.unique()) > 1:
            temp = pd.DataFrame(temp.fillna(temp.median()))
            lr = LR()
            lr.fit(temp, data[target])
            prob = lr.predict_proba(temp)[:, 1]
            auc = metrics.roc_auc_score(data[target], prob)
            if auc > auc_score:
                if silent:
                    print('-'.join(value), ' AUC (train): ', auc)
                new['-'.join(value)] = data[corr_list[index][0]] - data[corr_list[index][1]]

    return new


def simAnneal(train, predictors, target, param=param_1, results=True):
    """ Hyper parameter tuning with simulated annealing.

    Employes the simulated annealing to find the optimal hyper parameters and 
    return an optimized classifier.

    Args:
        train: A training set of your machine learning project.
        predictors: The predictors of your dataset.
        target: The target variable in your dataset; limited to binary.
        param: A hyper parameter dictionary for tuning task; default with param_1.
        results: Whether print the progress out; default with True.

    Returns:
        optimized_clf: An optimized classifier after hyper parameter tuning.
    """
    print('\nsimulating...')

    gbm = LGBMClassifier(
        learning_rate=0.01, n_estimators=5000, objective='binary', metric='auc',
        save_binary=True, is_unbalance=True, random_state=2017
    )

    sa = SimulatedAnneal(gbm, param, T=10.0, T_min=0.001, alpha=0.75,
                         verbose=True, max_iter=0.25, n_trans=5, max_runtime=300,
                         cv=3, scoring='roc_auc', refit=True)
    sa.fit(train[predictors].as_matrix(), train[target].as_matrix())

    if results:
        print('\n best score: ', sa.best_score_,
              '\n best parameters: ', sa.best_params_)
    optimized_clf = sa.best_estimator_

    return optimized_clf


def report(classifier, train, test, predictors, target):
    """ A general report.

    Prints model report with a classifier on training and test dataset.

    Args:
        classifier: A classifier to report.
        trian: A training set of your machine learning project.
        test: A test set of your machine learning project.
        predictors: The predictors of your dataset.
        target: The target variable in your dataset; limited to binary.
    """
    print('\npredicting...')
    dtrain_predictions = classifier.predict(train[predictors])
    dtest_predictions = classifier.predict(test[predictors])
    dtrain_predprob = classifier.predict_proba(train[predictors])[:, 1]
    dtest_predprob = classifier.predict_proba(test[predictors])[:, 1]

    print("\nModel Report")
    print("Accuracy : %f" % metrics.accuracy_score(
        train[target], dtrain_predictions))
    print("AUC Score (train): %f" %
          metrics.roc_auc_score(train[target], dtrain_predprob))
    print('AUC Score (test): %f' %
          metrics.roc_auc_score(test[target], dtest_predprob))
    print(classification_report(test[target], dtest_predictions))

    return None


def plot_confusion_matrix(cm, classes=[0, 1], normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """ Report confusion matrix plot.

    Prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`.
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


def report_CM(classifier, test, predictors, target):
    """ A report on confusion matrix.

    Reports the recall rate of the classifier on test data and plots out 
    confusion matrix.

    Args:
        classifier: A classifier to report.
        test: A test set of your machine learning project.
        predictors: The predictors of your dataset.
        target: The target variable in your dataset; limited to binary.
    """
    print('\npredicting...')
    dtest_predictions = classifier.predict(test[predictors])
    print("\nModel Report")
    cnf_matrix = confusion_matrix(test[target], dtest_predictions)
    np.set_printoptions(precision=2)
    print("Recall metric in the testing dataset: ", cnf_matrix[
        1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))
    plt.figure()
    plot_confusion_matrix(cnf_matrix)

    return None


def report_PR(classifier, test, predictors, target):
    """ A report on precision-recall curve.

    Reports precision-recall curve and gives average precision.

    Args:
        classifier: A classifier to report.
        test: A test set of your machine learning project.
        predictors: The predictors of your dataset.
        target: The target variable in your dataset; limited to binary.
    """
    print('\npredicting...')
    dtest_predprob = classifier.predict_proba(test[predictors])[:, 1]
    precision, recall, _ = precision_recall_curve(test[target], dtest_predprob)
    average_precision = average_precision_score(test[target], dtest_predprob)
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

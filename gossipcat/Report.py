#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
import itertools
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns 

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
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return None

class Report(object):

    def __init__(self, classifier, train, test, target, predictors, predict=False, is_sklearn=False):
        """
        Args:
            classifier: A classifier to report.
            train: A training set of your machine learning project.
            test: A test set of your machine learning project.
            target: The target variable; limited to binary.
            predictors: The predictors.
        """
        try:
            self.classifier = classifier
            self.train = train
            self.test = test
            self.target = target 
            self.predictors = predictors
            self.is_sklearn = is_sklearn
        except Exception as e:
            print("[INFO] NO modle and data initialized. You may need to specify teat labels and predictions.")
            self.classifier = []
            self.train = []
            self.test = []
            self.target = []
            self.predictors = []
            self.is_sklearn = is_sklearn

        if predict:
            self.Prection()
        else:
            self.train_predprob = []
            self.test_predprob = []
            self.train_predictions = []
            self.test_predictions = []

    def Prection(self):
        print('\npredicting...')
        if self.is_sklearn:
            self.train_predictions = self.classifier.predict(self.train[self.predictors])
            self.test_predictions = self.classifier.predict(self.test[self.predictors])
            self.train_predprob = self.classifier.predict_proba(self.train[self.predictors])[:, 1]
            self.test_predprob = self.classifier.predict_proba(self.test[self.predictors])[:, 1]
        else:
            self.train_predprob = self.classifier.predict(self.train[self.predictors])
            self.test_predprob = self.classifier.predict(self.test[self.predictors])
            self.train_predictions = np.where(self.train_predprob>=0.5, 1, 0)
            self.test_predictions = np.where(self.test_predprob>=0.5, 1, 0)
        print('\ndone.')
        return None

    def GN(self):
        """ A general report.

        Prints model report with a classifier on training and test dataset.
        """
        message = "\nModel Report"+\
                  "\nAccuracy: %0.3f" % metrics.accuracy_score(self.train[self.target], self.train_predictions)+\
                  "\nROC AUC Score (train): %0.3f" % metrics.roc_auc_score(self.train[self.target], self.train_predprob)+\
                  "\nROC AUC Score (test): %0.3f" % metrics.roc_auc_score(self.test[self.target], self.test_predprob)+\
                  "\nPR AUC Score (train): %0.3f" % metrics.average_precision_score(self.train[self.target], self.train_predprob)+\
                  "\nPR AUC Score (test): %0.3f" % metrics.average_precision_score(self.test[self.target], self.test_predprob)+\
                  "\n"+classification_report(self.test[self.target], self.test_predictions)
        print(message)
        return None

    def CM(self):
        """ A report on confusion matrix.

        Reports the recall rate of the classifier on test data and plots out 
        confusion matrix.
        """    
        print("\nModel Report")
        cnf_matrix = confusion_matrix(self.test[self.target], self.test_predictions)
        np.set_printoptions(precision=2)
        print("Recall metric in the testing dataset: ", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))

        plt.figure(figsize=(6, 6))
        plot_confusion_matrix(cnf_matrix)
        plt.show()
        return None

    def CAP(self):
        """ A report on Cumulative Accuracy Profile (CAP) curve.

        Reports CAP curve and gives accuracy ratio (AR).
        """
        df = pd.DataFrame({'label': self.test[self.target], 'prob': self.test_predprob})
        N = df.shape[0]
        N_pos = df[df['label']==1].shape[0]

        df = df.sort_values(by='prob', ascending=False).reset_index(drop=True)
        df = df.reset_index()

        df['alarm_rate'] = (df['index'] + 1) / N
        df['cum_hit'] = df['label'].cumsum()
        df['hit_rate'] = df['cum_hit'] / N_pos
        df['random'] = df['alarm_rate']
        df['perfect'] = df['index'].apply(lambda x: x/N_pos if x/N_pos < 1 else 1)
        del df['index']

        plt.figure(figsize=(6, 5.5))
        plt.step(x=df['alarm_rate'], y=df['hit_rate'])
        plt.step(x=df['alarm_rate'], y=df['random'], color='gray')
        plt.step(x=df['alarm_rate'], y=df['perfect'], color='green')

        accuracy_ratio = round(np.sum(df['hit_rate'] - df['random']) / np.sum(df['perfect'] - df['random']), 4)

        plt.title('Cumulative Accuracy Profile: AR={0:0.4f}'.format(accuracy_ratio))
        plt.xlabel('Alarm Rate')
        plt.ylabel('Hit Rate')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()
        return None

    def ROC(self):
        """ A report on Receiver Operating Charactoristic(ROC) curve.

        Reports ROC curve and gives roc auc score.
        """
        roc_auc = metrics.roc_auc_score(self.test[self.target], self.test_predprob)
        fpr, tpr, _ = metrics.roc_curve(self.test[self.target], self.test_predprob)

        plt.figure(figsize=(6, 5.5))
        plt.plot(fpr, tpr, label='Classifier (area = %.3f)'%roc_auc)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Charactoristic')
        plt.legend(loc='lower right')
        plt.show()
        return None

    def PR(self):
        """ A report on precision-recall curve.

        Reports precision-recall curve and gives average precision.
        """
        precision, recall, _ = precision_recall_curve(self.test[self.target], self.test_predprob)
        average_precision = average_precision_score(self.test[self.target], self.test_predprob)
        
        print('\nModel Report')
        print('Average Precision: {0:0.3f}'.format(average_precision))

        plt.figure(figsize=(6, 5.5))
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.5, color='red')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall curve: AP={0:0.3f}'.format(average_precision))
        plt.show()
        return None

    def ALL(self, is_lgb=False):
        """Include all methods.
        """
        self.GN()
        self.CM()
        self.CAP()
        self.ROC()
        self.PR()
        return None


class Visual(object):
    """docstring for Visual"""
    def __init__(self, test_target, test_predprob):
        super(Visual, self).__init__()
        self.test_target = test_target
        self.test_predprob = test_predprob

    def CM(self, threshold=0.5, normalize=False):
        """ A report on confusion matrix.

        Reports the recall rate of the classifier on test data and plots out 
        confusion matrix.
        """
        self.test_predictions = np.where(self.test_predprob>=threshold, 1, 0)
        print("\nModel Report")
        cnf_matrix = confusion_matrix(self.test_target, self.test_predictions)
        np.set_printoptions(precision=2)
        print("Recall metric in the testing dataset: ", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))
        print(classification_report(self.test_target, self.test_predictions))

        plt.figure(figsize=(6, 6))
        plot_confusion_matrix(cnf_matrix, normalize=normalize)
        plt.show()
        return None

    def CAP(self, alarm=0.05):
        """ A report on Cumulative Accuracy Profile (CAP) curve.

        Reports CAP curve and gives accuracy ratio (AR).

        Args:
            alarm: check hit rate at the alarm rate of, default at 0.05.
        """
        df = pd.DataFrame({'label': self.test_target, 'prob': self.test_predprob})
        N = df.shape[0]
        N_pos = df[df['label']==1].shape[0]

        df = df.sort_values(by='prob', ascending=False).reset_index(drop=True)
        df = df.reset_index()

        df['alarm_rate'] = (df['index'] + 1) / N
        df['cum_hit'] = df['label'].cumsum()
        df['hit_rate'] = df['cum_hit'] / N_pos
        df['random'] = df['alarm_rate']
        df['perfect'] = df['index'].apply(lambda x: x / N_pos if x / N_pos < 1 else 1)
        del df['index']
        self.df_cap = df 

        plt.figure(figsize=(6, 6))
        plt.step(x=df['alarm_rate'], y=df['perfect'], color='#2ca02c', label='perfect')
        plt.step(x=df['alarm_rate'], y=df['hit_rate'], color='#1f77b4', label='model')
        plt.step(x=df['alarm_rate'], y=df['random'], color='#7f7f7f', label='guess')

        accuracy_ratio = round(np.sum(df['hit_rate'] - df['random']) / np.sum(df['perfect'] - df['random']), 4)

        plt.title('Cumulative Accuracy Profile: AR={0:0.4f}'.format(accuracy_ratio))
        plt.xlabel('Alarm Rate')
        plt.ylabel('Hit Rate')
        plt.legend(loc='lower right', title='Models')
        plt.grid()

        if 0 < alarm < 1:
            from matplotlib.patches import Circle
            from matplotlib.patheffects import withStroke

            hitAtAlarm = df[df['alarm_rate']>=alarm][['hit_rate']].iloc[0]

            circle = Circle((alarm, hitAtAlarm), 0.05, clip_on=False, zorder=10, linewidth=1,
                            edgecolor='black', facecolor=(0, 0, 0, .0125),
                            path_effects=[withStroke(linewidth=5, foreground='w')])
            marker = plt.scatter(alarm, hitAtAlarm, s=300, c='red', marker='+', clip_on=False)

            plt.gcf().gca().add_artist(circle)
            plt.gcf().gca().add_artist(marker)
            plt.show()

            print('The hit rate at alarm rate of %.2f is: %.2f' %(alarm, hitAtAlarm))

        plt.show()
        return None

    def ROC(self):
        """ A report on Receiver Operating Charactoristic(ROC) curve.

        Reports ROC curve and gives roc auc score.
        """
        roc_auc = metrics.roc_auc_score(self.test_target, self.test_predprob)
        fpr, tpr, _ = metrics.roc_curve(self.test_target, self.test_predprob)

        plt.figure(figsize=(6, 5.5))
        plt.plot(fpr, tpr, label='Classifier (area = %.3f)'%roc_auc)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Charactoristic')
        plt.legend(loc='lower right')
        plt.show()
        return None

    def PR(self):
        """ A report on precision-recall curve.

        Reports precision-recall curve and gives average precision.
        """
        precision, recall, _ = precision_recall_curve(self.test_target, self.test_predprob)
        average_precision = average_precision_score(self.test_target, self.test_predprob)
        
        print('\nModel Report')
        print('Average Precision: {0:0.3f}'.format(average_precision))

        plt.figure(figsize=(6, 5.5))
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.5, color='red')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall curve: AP={0:0.3f}'.format(average_precision))
        plt.show()
        return None

class PSI(object):
    """Population Stability Index (PSI) Analysis between two samples
    
    Author:
        Ewen
        Matthew Burke
        github.com/mwburke
        worksofchart.com
    """
    def __init__(self, sample1, sample2):
        """
        Args:
            sample1: numpy matrix of original values
            sample1: numpy matrix of new values, same size as expected
        """
        super(PSI, self).__init__()
        self.sample1 = sample1
        self.sample2 = sample2
        self.buckets = None
        self.df = None
        self.PSI = None

    def calculate(self, buckettype='bins', buckets=10):
        """ Calculate PSI for two samples
        Args:
            buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
            buckets: number of quantiles to use in bucketing variables
        
        Returns:
            psi_values: psi values
        """
        self.buckets = buckets

        def scale_range(input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min + 0.0001) + 0.0001
            input += min
            return input

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(self.sample1), np.max(self.sample1))
        elif buckettype == 'quantiles':
            breakpoints = np.stack([np.percentile(self.sample1, b) for b in breakpoints])

        cnt_sample1 = np.histogram(self.sample1, breakpoints)[0]
        cnt_sample2 = np.histogram(self.sample2, breakpoints)[0]

        per_sample1 = cnt_sample1 / len(self.sample1)
        per_sample2 = cnt_sample2 / len(self.sample2)

        df = pd.DataFrame({'bucket': np.arange(1, self.buckets+1), 
                           'breakpoint': breakpoints[1:], 
                           'count_sample1': cnt_sample1, 
                           'count_sample2': cnt_sample2,
                           'percent_sample1': per_sample1, 
                           'percent_sample2': per_sample2})

        df['psi'] = (df['per_sample1'] - df['per_sample2']) * np.log((df['per_sample1'] / (df['per_sample2'] + 0.0001)))

        self.df = df
        self.PSI = round(np.sum(df['psi']), 5)
        return self.PSI

    def plot_density(self):
        """ Plot density plots of two samples.
        """
        sns.set_style('white')

        plot = sns.kdeplot(self.sample1, shade=True)
        plot = sns.kdeplot(self.sample2, shade=True)
        plot.set(yticklabels=[], xticklabels=[])
        sns.despine(left=True)
        return None

    def plot_histgram(self):
        """ Plot histgrams of two samples.
        """
        sns.set_style('white')

        percents = self.df[['per_sample1', 'per_sample2', 'bucket']]\
                          .melt(id_vars=['bucket'])\
                          .rename(columns={'variable': 'Population', 'value': 'Percent'})

        plot = sns.barplot(x='bucket', y='Percent', hue='Population', data=percents)
        plot.set(xlabel='Bucket', ylabel='Population Percent')
        sns.despine(left=True)
        return None

def performace_by_month(df, target, prob, num_months=7):
    from sklearn.metrics import average_precision_score
    def get_date_with_delta(delta):
        from datetime import date, timedelta
        return (date.today() - timedelta(days=delta)).strftime('%Y-%m-%d')

    df_score = pd.DataFrame()
    date_start_list = []
    date_end_list = []
    ap_list = []
    for m in reversed(range(9, num_months)):
        delta_start = 30*(m+1)
        delta_end = 30*m
        date_start = get_date_with_delta(delta_start)
        date_end = get_date_with_delta(delta_end)
        print(date_start)
        print(date_end)
        tmp = df[(df['udf_return_date'] > date_start) & (df['udf_return_date'] <= date_end)]
        date_start_list.append(date_start)
        date_end_list.append(date_end)
        ap_list.append(average_precision_score(tmp[target], tmp[prob]))
    df_score['data_start'] = date_start_list
    df_score['date_end'] = date_end_list
    df_score['average_precision'] = ap_list
    return df_score





        
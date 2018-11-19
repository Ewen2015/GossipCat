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
import lightgbm as lgb
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

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
    print(cm)

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
        return message

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

    def FI(self):
        """A report on feature importance.
        
        Reports feature importance of LightGBM models.
        """
        lgb.plot_importance(self.classifier, figsize=(8, 7))
        plt.show()
        if self.is_sklearn:
            pass
        else:
            print(pd.DataFrame(self.predictors, columns=['Feature']))
        return None

    def ALL(self, is_lgb=False):
        """Include all methods.
        """
        self.GN()
        self.CM()
        self.ROC()
        self.PR()
        if is_lgb:
            self.FI()
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


        
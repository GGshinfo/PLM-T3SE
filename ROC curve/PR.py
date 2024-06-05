import csv
import numpy as np
import pandas as pd
import math
import os
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import KFold, GridSearchCV, LeaveOneOut, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

def getMeanProba(indexFile, ProbaFile):
    GBM_index = pd.read_csv(indexFile, header=None)
    GBM_proba = pd.read_csv(ProbaFile, header=None)
    GBM_index = np.array(GBM_index)
    GBM_proba = np.array(GBM_proba)
    GBM_AUC = []
    r = GBM_index.shape[0]
    c = GBM_index.shape[1]
    mean_proba = []
    GBM_mean_proba1 = []

    for i in range(c):
        sum = 0
        for j in range(r):
            a = GBM_index[j]
            a = np.array(a)
            s = np.argwhere(a == i)
            sum += GBM_proba[j, s]
        mean_proba.extend(sum / 1)
    mean_proba = pd.DataFrame(mean_proba)
    mean_proba = np.array(mean_proba)
    for i in range(mean_proba.shape[0]):
        for j in range(mean_proba.shape[1]):
            GBM_mean_proba1.append(mean_proba[i][j])
    return GBM_mean_proba1

def plot_pr(precision, recall, pr_auc, label):
    plt.plot(recall, precision, color='darkorange', label='PR curve (AUC = %0.4f)' % pr_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(label)
    plt.legend(loc="best")


def main():
    train_y = pd.read_csv('train_lable.csv')
    train_y = np.array(train_y)
    lable_ls = train_y[:, 1]
    lable_ls = np.array(lable_ls)
    lable_ls = lable_ls.tolist()

    NB_MeanProba = getMeanProba('NBtest_index830D.csv', 'NBy_proba830D.csv')
    NB_fpr, NB_tpr, NB_thresholds = roc_curve(lable_ls, NB_MeanProba)
    NB_roc_auc = metrics.auc(NB_fpr, NB_tpr)
    print(NB_roc_auc)

    RF_MeanProba = getMeanProba('RFtest_index830D.csv', 'RFy_proba830D.csv')
    RF_fpr, RF_tpr, RF_thresholds = roc_curve(lable_ls, RF_MeanProba)
    RF_roc_auc = metrics.auc(RF_fpr, RF_tpr)
    print(RF_roc_auc)

    SVM_MeanProba = getMeanProba('SVMtest830D_index.csv', 'SVMy830D_proba.csv')
    SVM_fpr, SVM_tpr, SVM_thresholds = roc_curve(lable_ls, SVM_MeanProba)
    SVM_roc_auc = metrics.auc(SVM_fpr, SVM_tpr)
    print(SVM_roc_auc)


    GBM_MeanProba = getMeanProba('GBMtest830D_index.csv', 'GBMy830D_proba.csv')
    GBM_fpr, GBM_tpr, GBM_thresholds = roc_curve(lable_ls, GBM_MeanProba)
    GBM_roc_auc = metrics.auc(GBM_fpr, GBM_tpr)
    print(GBM_roc_auc)


    KNN_MeanProba = getMeanProba('KNNtest830D_index.csv', 'KNNy830D_proba.csv')
    KNN_fpr, KNN_tpr, KNN_thresholds = roc_curve(lable_ls, KNN_MeanProba)
    KNN_roc_auc = metrics.auc(KNN_fpr, KNN_tpr)
    print(KNN_roc_auc)

    MLP_MeanProba = getMeanProba('MLPtest830D_index.csv', 'MLPy830D_proba.csv')
    MLP_fpr, MLP_tpr, MLP_thresholds = roc_curve(lable_ls, MLP_MeanProba)
    MLP_roc_auc = metrics.auc(MLP_fpr, MLP_tpr)
    print(MLP_roc_auc)



    plt.figure()
    plt.plot(KNN_fpr, KNN_tpr, color='darkgreen',
             label='KNN (AUC = %0.4f)' %  KNN_roc_auc)
    plt.plot(SVM_fpr, SVM_tpr, color='darkorange',
             label='SVM (AUC = %0.4f)' %  SVM_roc_auc)
    plt.plot(RF_fpr, RF_tpr, color='darkviolet',
             label='RF (AUC = %0.4f)' %  RF_roc_auc)

    plt.plot(NB_fpr, NB_tpr, color='darkslateblue',
             label='NB (AUC = %0.4f)' %  NB_roc_auc)

    plt.plot(GBM_fpr, GBM_tpr, color='darkturquoise',
             label='GBM (AUC = %0.4f)' % GBM_roc_auc)

    plt.plot(MLP_fpr, MLP_tpr, color='darkred',
             label='MLP (AUC = %0.4f)' % MLP_roc_auc)

    plt.plot([0, 1], [0, 1], color='darkcyan', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate ')
    plt.ylabel('True Positive Rate ')
    plt.title('(a)', fontname='Times New Roman', fontsize=14, loc='center')
    plt.grid(False)
    plt.legend(loc="best")
    plt.show()
    plt.savefig('roc1.png', dpi=600)




main()






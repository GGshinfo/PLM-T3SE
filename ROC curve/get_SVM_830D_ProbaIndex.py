import csv
import sys
import numpy as np
import pandas as pd
import math
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import KFold, GridSearchCV, LeaveOneOut, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier


def main():

    train_x_NMBAC = pd.read_csv('train_830.csv')
    train_x_NMBAC = np.array(train_x_NMBAC)

    train_x = train_x_NMBAC
    print(train_x.shape)

    pro_y = pd.read_csv('train_lable.csv')
    pro_y = np.array(pro_y)
    pro_y = np.delete(pro_y, 0, axis=1)
    pro_y = pd.DataFrame(pro_y)
    train_y = pro_y.values.ravel()

    clf = SVC(C=8192, gamma=0.0001220703125, probability=True)

    print('10折:')
    cv = KFold(n_splits=10, shuffle=True)
    probass_y = []
    SVMtest_index = []
    for train, test in cv.split(train_x, train_y):
            SVMtest_index.extend(test)
            probas_ = clf.fit(train_x[train], train_y[train]).predict_proba(train_x[test])
            y_train_pred = clf.predict(train_x[test])
            y_train_probas = clf.predict_proba(train_x[test])
            probass_y.extend(y_train_probas[:, 1])
            auc = roc_auc_score(train_y[test], y_train_probas[:, 1])

    csvFile = open("SVMy830D_proba.csv", 'a', newline='')
    writer = csv.writer(csvFile, dialect='excel')
    writer.writerow(probass_y)
    csvFile.close()

    csvFile = open("SVMtest830D_index.csv", 'a', newline='')
    writer = csv.writer(csvFile, dialect='excel')
    writer.writerow(SVMtest_index)
    csvFile.close()

main()



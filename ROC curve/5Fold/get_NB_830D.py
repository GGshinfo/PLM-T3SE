import sys
import numpy as np
import pandas as pd
import math
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold, GridSearchCV, LeaveOneOut, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

def main():
    train_x_NMBAC = pd.read_csv('../train_830.csv')
    train_x_NMBAC = np.array(train_x_NMBAC)
    train_x = train_x_NMBAC
    print(train_x.shape)

    pro_y = pd.read_csv('../train_lable.csv')
    pro_y = np.array(pro_y)
    pro_y = np.delete(pro_y, 0, axis=1)
    pro_y = pd.DataFrame(pro_y)
    train_y = pro_y.values.ravel()

    acc = []
    sn = []
    sp = []
    f1 = []
    mcc = []
    clf = GaussianNB()
    cv = KFold(n_splits=10, shuffle=True)
    probass_y = []
    NBtest_index = []
    pred_y = []
    pro_5y = []
    for train, test in cv.split(train_x):
            x_train, x_test = train_x[train], train_x[test]
            y_train, y_test = train_y[train], train_y[test]
            NBtest_index.extend(test)
            probas_ = clf.fit(x_train, y_train).predict_proba(x_test)
            y_train_pred = clf.predict(x_test)
            y_train_probas = clf.predict_proba(x_test)
            probass_y.extend(y_train_probas[:, 1])
            pred_y.extend(y_train_pred)
            pro_5y.extend(y_test)
    cm = confusion_matrix(pro_5y, pred_y)
    tn, fp, fn, tp = cm.ravel()
    ACC = (tp + tn) / (tp + tn + fp + fn)
    SN = tp / (tp + fn)
    SP = tn / (tn + fp)
    PR = tp / (tp + fp)
    MCC = (tp * tn - fp * fn) / math.sqrt((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn))
    F1 = (2 * SN * PR) / (SN + PR)
    print('meanACC:', ACC)
    print('meanSN:', SN)
    print('meanSP:', SP)
    print('meanF1:', F1)
    print('meanMCC:', MCC)
    print('PR:', PR)


main()



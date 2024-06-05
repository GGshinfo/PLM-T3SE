import sys
import numpy as np
import pandas as pd
import math
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold, GridSearchCV, LeaveOneOut, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

def main():
    train_830 = pd.read_csv('../train_830.csv')
    train_x = np.array(train_830)
    print(train_x.shape)

    test_830 = pd.read_csv('../test_830.csv')
    test_x = np.array(test_830)
    print(test_x.shape)

    train_y = pd.read_csv('../train_lable.csv')
    train_y = np.array(train_y)
    train_y = np.delete(train_y, 0, axis=1)
    train_y = pd.DataFrame(train_y)
    train_y = train_y.values.ravel()


    test_y = pd.read_csv('../test_lable.csv')
    test_y = np.array(test_y)
    test_y = np.delete(test_y, 0, axis=1)
    test_y = pd.DataFrame(test_y)
    test_y = test_y.values.ravel()

    acc = []
    sn = []
    sp = []
    f1 = []
    mcc = []
    CC = []
    gammas = []
    for i in range(-5, 15, 2):
        CC.append(2 ** i)
    for i in range(3, -15, -2):
        gammas.append(2 ** i)
    param_grid = {"C": CC, "gamma": gammas}
    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    gs = GridSearchCV(SVC(probability=True), param_grid, cv=kf)  # 网格搜索
    gs.fit(train_x, train_y)
    print(gs.best_estimator_)
    ''''''
    print(gs.best_score_)
    ''''''
    clf = gs.best_estimator_
    print('10折结果：')
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
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    ACC = (tp + tn) / (tp + tn + fp + fn)
    SN = tp / (tp + fn)
    SP = tn / (tn + fp)
    PR = tp / (tp + fp)
    MCC = (tp * tn - fp * fn) / math.sqrt((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn))
    F1 = (2 * SN * PR) / (SN + PR)
    print('meanACC:', ACC)
    print('meanSN:', SN)
    print('meanSP:',SP )
    print('meanF1:', F1)
    print('meanMCC:', MCC)
    print('PR:', PR)
main()



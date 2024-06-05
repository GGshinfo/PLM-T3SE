import math
import pickle
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn import datasets
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold, GridSearchCV
from xgboost import XGBClassifier

# 读训练集数据
with open("../feature_t3se/T5/train_positive_t5.pkl", "rb") as tf:
    feature_dict = pickle.load(tf)
prot_train_P = np.array([item for item in feature_dict.values()])
prot_train_P = np.insert(prot_train_P, 0, values=[0 for _ in range(prot_train_P.shape[0])], axis=1)
print("prot_train_P:", prot_train_P.shape)
with open("../feature_t3se/T5/train_negative_t5.pkl", "rb") as tf:
    feature_dict = pickle.load(tf)
prot_train_N = np.array([item for item in feature_dict.values()])
prot_train_N = np.insert(prot_train_N, 0, values=[-1 for _ in range(prot_train_N.shape[0])], axis=1)
print("train_N:", prot_train_N.shape)
prot_train = np.row_stack((prot_train_P, prot_train_N))
print("prot_train:", prot_train.shape)

# 划分特征与标签
prot_Y, prot_X = prot_train[:, 0], prot_train[:, 1:]
print("prot标签:", prot_Y.shape)  # 标签
print("prot特征:", prot_X.shape)  # 特征
# 划分数据集
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)  # 按照比例划分数据集为训练集与测试集

train_x_AAC_DPC = pd.read_csv('../feature_t3se/PSSM-AAC-DPC/trainData_pssm_AAC_DPC.csv')
train_x_AAC_DPC = np.array(train_x_AAC_DPC)
train_x_AAC_DPC = np.delete(train_x_AAC_DPC, 0, axis=1)

test_x_AAC_DPC = pd.read_csv('../feature_t3se/PSSM-AAC-DPC/testData_pssm_AAC_DPC.csv')
test_x_AAC_DPC = np.array(test_x_AAC_DPC)
test_x_AAC_DPC = np.delete(test_x_AAC_DPC, 0, axis=1)



std = StandardScaler()
train_x_AAC_DPC = std.fit_transform(train_x_AAC_DPC)
test_x_AAC_DPC = std.fit_transform(test_x_AAC_DPC)


train_x = np.hstack((prot_X, train_x_AAC_DPC))
print("train_x: ", train_x.shape)



#独立集
with open("../feature_t3se/T5/test_positive_t5.pkl", "rb") as tf:
    feature_dict = pickle.load(tf)
prot_test_P = np.array([item for item in feature_dict.values()])
prot_test_P = np.insert(prot_test_P, 0, values=[0 for _ in range(prot_test_P.shape[0])], axis=1)
print("prot_test_P:", prot_test_P.shape)
with open("../feature_t3se/T5/test_negative_t5.pkl", "rb") as tf:
    feature_dict = pickle.load(tf)
prot_test_N = np.array([item for item in feature_dict.values()])
prot_test_N = np.insert(prot_test_N, 0, values=[-1 for _ in range(prot_test_N.shape[0])], axis=1)
print("prot_test_N:", prot_test_N.shape)
prot_test = np.row_stack((prot_test_P, prot_test_N))
print("prot_test:", prot_test.shape)

# 划分特征与标签
lable_test, data_test = prot_test[:, 0], prot_test[:, 1:]
data_test = np.hstack((data_test, test_x_AAC_DPC))
print("lable_test:", lable_test.shape)
print(("data_test:", data_test.shape))

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
lable_train = le.fit_transform(prot_Y)
lable_test = le.fit_transform(lable_test)

# 分类器模型
model = XGBClassifier(random_state=1)
model.fit(train_x, lable_train)
# 特征重要性
import_level = model.feature_importances_
index = np.argsort(import_level)[::-1]
rank_matrix = np.zeros((train_x.shape[0], train_x.shape[1]))
rank_matrix_test = np.zeros((data_test.shape[0], data_test.shape[1]))

for i in range(train_x.shape[1]):
    rank_matrix[:, i] = train_x[:, index[i]]
    rank_matrix_test[:, i] = data_test[:, index[i]]
fold10score = []
inde = []
w = train_x.shape[1]
print(w)
for lag in range(100, w, 20):
    print('特征个数为%d个时所有的指标情况' % lag)
    # 模型构建
    mlp_clf__tuned_parameters = {"hidden_layer_sizes": [(32, 64), (64, 128), (16, 32)],
                                 "solver": ['adam', 'sgd', 'lbfgs'],
                                 "activation": ['logistic', 'tanh', 'relu'],
                                 "alpha": [0.0001, 0.001, 0.01],
                                 "learning_rate": ['constant', 'adaptive', 'invscaling']
                                 }
    mlp = MLPClassifier(max_iter=10000, n_iter_no_change=30)
    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    gs = GridSearchCV(mlp, mlp_clf__tuned_parameters, cv=kf)

    xtr = rank_matrix[:, 0:lag]
    gs.fit(xtr, lable_train)

    print('estimator:', gs.best_estimator_)
    print('score:', gs.best_score_)
    # 分类器取最优超参数
    clf = gs.best_estimator_

    probass_y = []
    test_index = []
    pred_y = []
    pro_10y = []
    cv = KFold(n_splits=10, shuffle=True)
    for train, test in cv.split(xtr):
        x_train, x_test = xtr[train], xtr[test]
        y_train, y_test = lable_train[train], lable_train[test]
        test_index.extend(test)
        clf.fit(x_train, y_train)
        y_test_pred = clf.predict(x_test)
        pred_y.extend(y_test_pred)
        pro_10y.extend(y_test)

    cm = confusion_matrix(pro_10y, pred_y)
    TN, FP, FN, TP = cm.ravel()
    ACC = (TP + TN) / (TP + TN + FP + FN)
    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN))
    Pre = TP / (TP + FP)
    F1 = (2 * SN * Pre) / (SN + Pre)
    print("10折后：")
    print('ACC：', ACC)
    print('SN:', SN)
    print('SP:', SP)
    print('MCC:', MCC)
    print('Pre:', Pre)
    print('F1:', F1)
    fold10score.append(ACC)


    pro_y_pred = clf.predict(rank_matrix_test[:, 0:lag])
    acc = accuracy_score(lable_test, pro_y_pred)
    inde.append(acc)
    cm = confusion_matrix(lable_test, pro_y_pred)
    testTN, testFP, testFN, testTP = cm.ravel()
    testACC = (testTP + testTN) / (testTP + testTN + testFP + testFN)
    testSN = testTP / (testTP + testFN)
    testSP = testTN / (testTN + testFP)
    testMCC = (testTP * testTN - testFP * testFN) / math.sqrt((testTP + testFN) * (testTP + testFP) * (testTN + testFP) * (testTN + testFN))
    testPre = testTP / (testTP + testFP)
    testF1 = (2 * testSN * testPre) / (testSN + testPre)
    print("独立测试集：")
    print('Acc:', testACC)
    print('SN:', testSN)
    print('SP:', testSP)
    print('MCC:', testMCC)
    print('Pre:', testPre)
    print('F1:', testF1)
print("10折准确率：", fold10score)
print("测试集准确率：", inde)





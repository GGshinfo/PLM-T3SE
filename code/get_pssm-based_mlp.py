import math
import pickle
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


print("PSSM-based-mlp:")
# 读训练集数据
train_y = pd.read_csv('../feature_t3se/PSSM/train_lable.csv')
train_y = np.array(train_y)
train_y = np.delete(train_y, 0, axis=1)
train_y = pd.DataFrame(train_y).values.ravel()

#PSSM-AAC
train_x_AAC = pd.read_csv('../feature_t3se/PSSM/trainData_pssm_AAC.csv')
train_x_AAC = np.array(train_x_AAC)
train_x_AAC = np.delete(train_x_AAC, 0, axis=1)
std = StandardScaler()
train_x_AAC = std.fit_transform(train_x_AAC)
print("train_x:", train_x_AAC.shape)

# 模型构建

mlp_clf__tuned_parameters = {"hidden_layer_sizes": [(32,64),(64,128),(16,32)],
                                 "solver": ['adam', 'sgd', 'lbfgs'],
                                 "activation": ['logistic', 'tanh', 'relu'],
                                 "alpha": [0.0001, 0.001, 0.01],
                                 "learning_rate":['constant', 'adaptive', 'invscaling']
                                 }
mlp = MLPClassifier(max_iter=10000, n_iter_no_change=30)
kf = KFold(n_splits=5, shuffle=True, random_state=123)
gs = GridSearchCV(mlp, mlp_clf__tuned_parameters, cv = kf)
gs.fit(train_x_AAC, train_y)

#  y是liable  xshi da
print('estimator:', gs.best_estimator_)
print('score:', gs.best_score_)


clf = gs.best_estimator_



Acc = []
Sen = []
Spe = []
Mcc = []
PR = []
FF1 = []
#10折交叉检验
print('10折交叉检验......')
cv = KFold(n_splits=10, shuffle=True)

NBtest_index = []
for train, test in cv.split(train_x_AAC):  # train test  是下标
    x_train, x_test = train_x_AAC[train], train_x_AAC[test]
    y_train, y_test = train_y[train], train_y[test]
    NBtest_index.extend(test)
    clf.fit(x_train, y_train)
    y_test_pre = clf.predict(x_test)
    cm = confusion_matrix(y_test, y_test_pre)
    # print(cm)
    TN, FP, FN, TP  = cm.ravel()
    ACC = (TP + TN) / (TP + TN + FP + FN)
    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN))
    Pre = TP / (TP + FP)
    F1 = (2 * SN * Pre) / (SN + Pre)
    Acc.append(ACC)
    Sen.append(SN)
    Spe.append(SP)
    Mcc.append(MCC)
    PR.append(Pre)
    FF1.append(F1)

print("10折结果：")
print("Acc:", Acc)
print("Sen:", Sen)
print("Spe:", Spe)
print("Mcc:", Mcc)
print("PR:", PR)
print("FF1:", FF1)

print('指标均值:')
print('meanAcc:', np.mean(Acc))
print('meanSen:', np.mean(Sen))
print('meanSpe:', np.mean(Spe))
print('meanMcc:', np.mean(Mcc))
print('meanPR:', np.mean(PR))
print('meanFF1:', np.mean(FF1))





#独立集检验
test_y = pd.read_csv('../feature_t3se/PSSM/test_lable.csv')
test_y = np.array(test_y)
test_y = np.delete(test_y, 0, axis=1)
test_y = pd.DataFrame(test_y).values.ravel()
#PSSM-AAC
test_x_AAC = pd.read_csv('../feature_t3se/PSSM/testData_pssm_AAC.csv')
test_x_AAC = np.array(test_x_AAC)
test_x_AAC = np.delete(test_x_AAC, 0, axis=1)
std = StandardScaler()
test_x_AAC = std.fit_transform(test_x_AAC)


lable_test_pre = clf.predict(test_x_AAC)
cm = confusion_matrix(test_y, lable_test_pre)
TN, FP, FN, TP  = cm.ravel()
ACC = (TP + TN) / (TP + TN + FP + FN)
SN = TP / (TP + FN)
SP = TN / (TN + FP)
MCC = (TP * TN - FP * FN) / math.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN))
Pre = TP / (TP + FP)
F1 = (2 * SN * Pre) / (SN + Pre)
print('Acc:', ACC)
print('SN:', SN)
print('SP:', SP)
print('MCC:', MCC)
print('Pre:', Pre)
print('F1:', F1)



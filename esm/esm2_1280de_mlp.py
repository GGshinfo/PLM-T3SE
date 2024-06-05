import math
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn import datasets
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold, GridSearchCV

# 读训练集数据
train_esm_data = pd.read_csv('t3_train_esm2_t33_650M_UR50D.csv')
values = train_esm_data.iloc[:, 1]
X = values.str.replace('[', '').str.replace(']','').str.split(',', expand=True)
X = np.array(X)
print("训练特征维度:", X.shape)  # 特征
train_Y = pd.read_csv('train_tag_1455.csv')
Y = train_Y.iloc[:, 1]
print("训练标签:", Y.shape)  # 标签

#模型构建
mlp_clf__tuned_parameters = {"hidden_layer_sizes": [(32,64),(64,128),(16,32)],
                                 "solver": ['adam', 'sgd', 'lbfgs'],
                                 "activation": ['logistic', 'tanh', 'relu'],
                                 "alpha": [0.0001, 0.001, 0.01],
                                 "learning_rate":['constant', 'adaptive', 'invscaling']
                                 }
mlp = MLPClassifier(max_iter=10000, n_iter_no_change=30)
kf = KFold(n_splits=5, shuffle=True, random_state=123)
gs = GridSearchCV(mlp, mlp_clf__tuned_parameters, cv = kf)
gs.fit(X, Y)
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
for train, test in cv.split(X):  # train test  是下标
    x_train, x_test = X[train], X[test]
    y_train, y_test = Y[train], Y[test]
    NBtest_index.extend(test)
    clf.fit(x_train, y_train)
    y_test_pre = clf.predict(x_test)
    cm = confusion_matrix(y_test, y_test_pre)
    print(cm)
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
# 读训练集数据
print("测试集结果：")
# 读训练集数据
test_esm_data = pd.read_csv('t3_test_esm2_t33_650M_UR50D.csv')
test_values = test_esm_data.iloc[:, 1]
X_test =test_values.str.replace('[', '').str.replace(']','').str.split(',', expand=True)
print("测试特征维度:", X_test.shape)  # 特征
Y_test = pd.read_csv('test_tag_216.csv')
Y_test = Y_test.iloc[:, 1]
print("测试标签:", Y_test.shape)  # 标签


Y_test_pre = clf.predict(X_test)
cm = confusion_matrix(Y_test, Y_test_pre)
print("cm:")
print(cm)
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






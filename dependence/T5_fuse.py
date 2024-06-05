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


print("PSSM-AAC-DPC-T5-SVM:")
# 读训练集数据
with open("../feature_t3se/T5/train_positive_t5.pkl", "rb") as tf:
    feature_dict = pickle.load(tf)
prot_train_P = np.array([item for item in feature_dict.values()])
prot_train_P = np.insert(prot_train_P, 0, values=[1 for _ in range(prot_train_P.shape[0])], axis=1)
print("prot_train_P:", prot_train_P.shape)
with open("../feature_t3se/T5/train_negative_t5.pkl", "rb") as tf:
    feature_dict = pickle.load(tf)
prot_train_N = np.array([item for item in feature_dict.values()])
prot_train_N = np.insert(prot_train_N, 0, values=[-1 for _ in range(prot_train_N.shape[0])], axis=1)
print("prot_train_N:", prot_train_N.shape)
prot_train = np.row_stack((prot_train_P, prot_train_N))
print("prot_train:", prot_train.shape)

# 划分特征与标签
prot_Y, prot_X = prot_train[:, 0], prot_train[:, 1:]
print("prot标签:", prot_Y.shape)  # 标签
print("prot特征:", prot_X.shape)  # 特征


#AAC_DPC
train_x_AAC_DPC = pd.read_csv('../feature_t3se/PSSM-AAC-DPC/trainData_pssm_AAC_DPC.csv')
train_x_AAC_DPC = np.array(train_x_AAC_DPC)
train_x_AAC_DPC = np.delete(train_x_AAC_DPC, 0, axis=1)
std = StandardScaler()
train_x_AAC_DPC = std.fit_transform(train_x_AAC_DPC)
train_x = np.hstack((prot_X, train_x_AAC_DPC))
print("train_x:", train_x.shape)

train_y = prot_Y

# 模型构建
# c和g为svm的参数
C = []
gamma = []
best_score = 0
# svm超参数调优
for i in range(-5, 15, 2):
    C.append(2 ** i)
for i in range(3, -15, -2):
    gamma.append(2 ** i)
param_grid = {"C": C, "gamma": gamma}
kf = KFold(n_splits=5, shuffle=True, random_state=123)
gs = GridSearchCV(SVC(probability=True), param_grid, cv=kf)  # 网格搜索
gs.fit(train_x, train_y)
print('estimator:', gs.best_estimator_)
print('score:', gs.best_score_)
# 分类器取最优超参数
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
for train, test in cv.split(train_x):  # train test  是下标
    x_train, x_test = train_x[train], train_x[test]
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
# 读训练集数据
with open("../feature_t3se/T5/test_positive_t5.pkl", "rb") as tf:
    feature_dict = pickle.load(tf)
prot_test_P = np.array([item for item in feature_dict.values()])
prot_test_P = np.insert(prot_test_P, 0, values=[1 for _ in range(prot_test_P.shape[0])], axis=1)
print("prot_test_P:", prot_test_P.shape)
with open("../feature_t3se/T5/test_negative_t5.pkl", "rb") as tf:
    feature_dict = pickle.load(tf)
prot_test_N = np.array([item for item in feature_dict.values()])
prot_test_N = np.insert(prot_test_N, 0, values=[-1 for _ in range(prot_test_N.shape[0])], axis=1)
print("prot_test_N:", prot_test_N.shape)
prot_test = np.row_stack((prot_test_P, prot_test_N))
print("prot_test:", prot_test.shape)



test_x_AAC_DPC = pd.read_csv('../feature_t3se/PSSM-AAC-DPC/testData_pssm_AAC_DPC.csv')
test_x_AAC_DPC = np.array(test_x_AAC_DPC)
test_x_AAC_DPC = np.delete(test_x_AAC_DPC, 0, axis=1)
test_x_AAC_DPC = std.fit_transform(test_x_AAC_DPC)

# 划分特征与标签
lable_test, data_test = prot_test[:, 0], prot_test[:, 1:]
data_test = np.hstack((data_test, test_x_AAC_DPC))


lable_test_pre = clf.predict(data_test)
cm = confusion_matrix(lable_test, lable_test_pre)
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









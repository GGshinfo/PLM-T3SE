import math
import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn import datasets
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold, GridSearchCV

# 读训练集数据
with open("../feature_t3se/T5/train_positive_t5.pkl", "rb") as tf:
    feature_dict = pickle.load(tf)
train_P = np.array([item for item in feature_dict.values()])
train_P = np.insert(train_P, 0, values=[1 for _ in range(train_P.shape[0])], axis=1)
print("train_P:", train_P.shape)
with open("../feature_t3se/T5/train_negative_t5.pkl", "rb") as tf:
    feature_dict = pickle.load(tf)
train_N = np.array([item for item in feature_dict.values()])
train_N = np.insert(train_N, 0, values=[-1 for _ in range(train_N.shape[0])], axis=1)
print("train_N:", train_N.shape)
train = np.row_stack((train_P, train_N))
print("train:", train.shape)

# 划分特征与标签
Y, X = train[:, 0], train[:, 1:]
print("标签:", Y.shape)  # 标签
print("特征:", X.shape)  # 特征


mlp = MLPClassifier(activation='tanh', alpha=0.01, hidden_layer_sizes=(32, 64),
              learning_rate='adaptive', max_iter=10000, n_iter_no_change=30,
              solver='lbfgs')

clf = mlp
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
print("测试集结果：")
with open("../feature_t3se/T5/test_positive_t5.pkl", "rb") as tf:
    feature_dict = pickle.load(tf)
test_P = np.array([item for item in feature_dict.values()])
test_P = np.insert(test_P, 0, values=[1 for _ in range(test_P.shape[0])], axis=1)
print("test_P:", test_P.shape)
with open("../feature_t3se/T5/test_negative_t5.pkl", "rb") as tf:
    feature_dict = pickle.load(tf)
test_N = np.array([item for item in feature_dict.values()])
test_N = np.insert(test_N, 0, values=[-1 for _ in range(test_N.shape[0])], axis=1)
print("test_N:", test_N.shape)
test = np.row_stack((test_P, test_N))
print("test:", test.shape)

# 划分特征与标签
Y_test, X_test = test[:, 0], test[:, 1:]
print("Y_test:" , Y_test.shape)
print("X_test:" , X_test.shape)

Y_test_pre = clf.predict(X_test)
cm = confusion_matrix(Y_test, Y_test_pre)
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


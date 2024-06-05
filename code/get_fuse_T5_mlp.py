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


print("PSSM-AAC-DPC-T5-mlp:")
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

mlp_clf__tuned_parameters = {"hidden_layer_sizes": [(32,64),(64,128),(16,32)],
                                 "solver": ['adam', 'sgd', 'lbfgs'],
                                 "activation": ['logistic', 'tanh', 'relu'],
                                 "alpha": [0.0001, 0.001, 0.01],
                                 "learning_rate":['constant', 'adaptive', 'invscaling']
                                 }
mlp = MLPClassifier(max_iter=10000, n_iter_no_change=30)
kf = KFold(n_splits=5, shuffle=True, random_state=123)
gs = GridSearchCV(mlp, mlp_clf__tuned_parameters, cv = kf)
gs.fit(train_x, train_y)

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

















#
# Acc = []
# Sen = []
# Spe = []
# Mcc = []
# 10次五折交叉检验
# for i in range(10):
#     print('第%d次五折正在进行......' % i)
#     cv = KFold(n_splits=5, shuffle=True)
#     proba_y = []
#     NBtest_index = []
#     pre_y = []
#     pro_y1 = []
#     for train, test in cv.split(X):  # train test  是下标
#         x_train, x_test = X[train], X[test]
#         y_train, y_test = Y[train], Y[test]
#         NBtest_index.extend(test)
#         proba_ = clf.fit(x_train, y_train).predict_proba(x_test)
#         y_train_pre = clf.predict(x_test)
#         y_train_proba = clf.predict_proba(x_test)
#         proba_y.extend(y_train_proba[:, 1])
#         pre_y.extend(y_train_pre)
#         pro_y1.extend(y_test)
#         cm = confusion_matrix(pro_y1, pre_y)
#         # print(cm)
#         TN, FP, FN, TP  = cm.ravel()
#         ACC = (TP + TN) / (TP + TN + FP + FN)
#         SN = TP / (TP + FN)
#         SP = TN / (TN + FP)
#         MCC = (TP * TN - FP * FN) / math.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN))
#         Acc.append(ACC)
#         Sen.append(SN)
#         Spe.append(SP)
#         Mcc.append(MCC)
# print('指标均值:')
# print('meanAcc:', np.mean(Acc))
# print('meanSen:', np.mean(Sen))
# print('meanSpe:', np.mean(Spe))
# print('meanMcc:', np.mean(Mcc))
# print('指标标准差:')
# print('stdAcc:', np.std(Acc))
# print('stdSen:', np.std(Sen))
# print('stdSpe:', np.std(Spe))
# print('stdMcc:', np.std(Mcc))

# 计算auc值
# auc = roc_auc_score(y_test,clf.predict_proba(x_test)[:, 1])
# fpr,tpr, thresholds = roc_curve(y_test,clf.decision_function(x_test))
#
# # 绘制ROC曲线
# plt.plot(fpr,tpr,color='darkred',label='SVM (AUC = %0.3f)' % auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([-0.02, 1.02])
# plt.ylim([-0.02, 1.02])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend(loc="lower right")
# plt.savefig('roc_curve.jpg',dpi=800)
# plt.show()






# # 独立集检验
# with open("D:/ProtTrans-master/positive_93.txt_t5.pkl","rb") as tf:
#     feature_dict = pickle.load(tf)
# test_P = np.array([item for item in feature_dict.values()])
# test_P = np.insert(test_P, 0, values=[1 for _ in range(test_P.shape[0])], axis=1)
# print("test_P:", test_P.shape)
# with open("D:/ProtTrans-master/negative_93.txt_t5.pkl","rb") as tf:
#     feature_dict = pickle.load(tf)
# test_N = np.array([item for item in feature_dict.values()])
# test_N = np.insert(test_N, 0, values=[-1 for _ in range(test_N.shape[0])], axis=1)
# print("test_N:", test_N.shape)
# test = np.row_stack((test_P, test_N))
# print("test:", test.shape)
# # 划分特征与标签
# Y, X = test[:, 0], test[:, 1:]
# print("标签:", Y.shape)  # 标签
# print("特征:", X.shape)  # 特征
# # 划分数据集
# X_train, X_test, y_train, y_test = train_test_split(
#     X, Y, test_size=0.2)  # 按照比例划分数据集为训练集与测试集
# # 模型构建
# clf = SVC(kernel='linear', C=1.0, probability=True)  # 创建SVM训练模型
# clf.fit(X_train, y_train)  # 对训练集数据进行训练
# clf_y_predict = clf.predict(X_test)  # 通过测试数据，得到测试标签
# scores = clf.score(X_test, y_test)  # 测试结果打分
# print("score:", scores)



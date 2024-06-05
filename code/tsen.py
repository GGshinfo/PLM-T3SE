import math
import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn import datasets
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold, GridSearchCV
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

def get_T5_feature():
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
    return prot_X, prot_Y


def get_PSSM_feature():
    # AAC_DPC
    train_x_AAC_DPC = pd.read_csv('../feature_t3se/PSSM-AAC-DPC/trainData_pssm_AAC_DPC.csv')
    train_x_AAC_DPC = np.array(train_x_AAC_DPC)
    train_x_AAC_DPC = np.delete(train_x_AAC_DPC, 0, axis=1)
    X = train_x_AAC_DPC
    return X


#将txt格式蛋白质序列提取出来，加到列表中
def get_fasta_list(file):
    f = open(file) #打开文件
    fastaList = []
    lines = f.readlines() #以行为单位读  lines里是以行为单位的列表
    L = len(lines) #获取行数
    for i in range(0, L):
        if i % 2 == 1:  #奇数行  txt格式数据中，奇数行是蛋白质序列
            lines[i] = lines[i].strip()#去除空格
            fastaList.append(lines[i]) #添加到列表
    return fastaList

amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
def one_hot_encode(sequence):
    # 创建一个全零矩阵，行数为氨基酸个数，列数为序列长度
    encoded_seq = np.zeros((len(amino_acids), len(sequence)))
    # 遍历序列中的每个氨基酸
    for i, amino_acid in enumerate(sequence):
        # 将对应氨基酸的位置设置为1
        encoded_seq[amino_acids.index(amino_acid), i] = 1
    return encoded_seq

def get_sequence_feature(file):
    sequencelist = get_fasta_list(file)  # 清洗数据，得到蛋白质列表
    # 转化为One-Hot编码
    encoded_sequences = []
    for sequence in sequencelist:
        encoded_seq = one_hot_encode(sequence)
        encoded_sequences.append(encoded_seq)
    # 将编码特征合并为一个矩阵
    encoded_matrix = np.concatenate(encoded_sequences, axis=1)
    return encoded_matrix

X, Y = get_T5_feature()
X1 = get_PSSM_feature()
Y1=Y
#encoded_matrix = get_sequence_feature('../data_t3se/train_1491/allTrain.txt')
tsne = TSNE(n_components=2, verbose=0, perplexity=50, learning_rate='auto', n_iter=5000, random_state=123)
z = tsne.fit_transform(X)
df =  pd.DataFrame()
df["Dimension-1"] = z[:,0]
df["Dimension-2"] = z[:,1]
y_new_lable = []
for i in Y:
    if i == 1:
        y_new_lable.append('T3SEs')
    if i == -1:
        y_new_lable.append('Non-T3SEs')
df["y"] = y_new_lable

#pssm
z1 = tsne.fit_transform(X1)
df1 = pd.DataFrame()
df1["Dimension-1"] = z1[:,0]
df1["Dimension-2"] = z1[:,1]
y_new_lable1 = []
for i in Y1:
    if i == 1:
        y_new_lable1.append('T3SEs')
    if i == -1:
        y_new_lable1.append('Non-T3SEs')
df1["y"] = y_new_lable1

# 创建子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))


# 绘制第二个子图
ax1.set_title('(a)', fontname='Times New Roman', fontsize=14)
graph1 = sns.scatterplot(data=df1, x='Dimension-1', y='Dimension-2', hue=y_new_lable1,
                        palette='BrBG_r', legend='full', ax=ax1)

# 绘制第一个子图
ax2.set_title('(b)', fontname='Times New Roman', fontsize=14)
graph2 = sns.scatterplot(data=df, x='Dimension-1', y='Dimension-2', hue=y_new_lable,
                        palette='BrBG_r', legend='full', ax=ax2)

# 保存图像
plt.savefig('tsne_all.png', dpi=600)


#
# graph = sns.scatterplot(data=df, x='Dimension-1', y='Dimension-2', hue=y_new_lable,
#                         palette='BrBG_r', legend='full')
# graph.set_title('(b)', fontname = 'Times New Roman', fontsize=14, loc='center')
# graph_for_output = graph.get_figure()
# graph_for_output.savefig('tsne_PSSM.png', dpi=600)


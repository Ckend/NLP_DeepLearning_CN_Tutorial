import bayes
from data_helpers import *
from sklearn.externals import joblib

posFile = "./data/train_food.txt"
negFile = "./data/train_notfood.txt"

print("正在获取训练矩阵及其分类向量")
trainList,classVec = loadTrainDataset(posFile,negFile)

print("正在将训练矩阵分词，并生成词表")
vectorized, vocabulary = jieba_cut_and_save_file(trainList,True)

bayes = bayes.oldNB(vocabulary)
# 初始化模型

print("正在训练模型")
bayes.train(vectorized,classVec)
# 训练

print("保存模型")
joblib.dump(bayes, "./arguments/train_model.m")

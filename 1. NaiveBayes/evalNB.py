import bayes
from data_helpers import *
from sklearn.externals import joblib

posFile = "./data/eval_food.txt"
negFile = "./data/eval_notfood.txt"

print("正在获取训练矩阵及其分类向量")
trainList,classVec = loadTrainDataset(posFile,negFile)

nb = joblib.load("arguments/train_model.m")
# 读取模型

results = []
for i in trainList:
    result = nb.classify_news(i)
    results.append(result)
# 测试模型

acc = 0.0
correct = 0
for i in range(len(classVec)):
    if results[i] == classVec[i]:
        correct += 1
acc = correct/len(classVec)
print("正确率为："+str(acc))
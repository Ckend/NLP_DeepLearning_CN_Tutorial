# NLP深度学习 中文教程

从最基本的机器学习模型朴素贝叶斯，介绍到双向神经网络BRNN.

博客(幻象客)网址:   https://huanxiangke.com

模型应用于极致安食网:   https://jizhianshi.com

## 1. 朴素贝叶斯分类器
各个目录和文件介绍：

    arguments: 存放保存下来的模型

    data: 存放训练和测试的文件

    bayes.py: 朴素贝叶斯模型，内含手写的朴素贝叶斯和scikit版的朴素贝叶斯

    data_helpers.py: 一些数据帮助函数

    evalNB.py: 测试文件

    trainNB.py: 训练文件


以3352条食品安全新闻和14946条非食品安全新闻为训练集
测试600:600的新闻集，准确率为97.7%

教程地址：https://huanxiangke.com/tutorials_dnn/naive_bayes_classify

## 2. 改进朴素贝叶斯分类器

重点：修改了jieba_cut_and_save_file函数，增加了权重接口，可以照葫芦画瓢增加其他词性的权重。

准确率为98.2%

教程地址：https://huanxiangke.com/blog/post/weights-improved-naive-bayes
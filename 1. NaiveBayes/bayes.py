# coding=utf-8
from numpy import *
from data_helpers import vectorize_newslist

class oldNB:
    """
    贝叶斯分类器, 自主编写
    """

    def __init__(self, vocabulary):
        self.p1V = None
        self.p0V = None
        self.p1 = None
        self.vocabulary = vocabulary

    def train(self, trainMatrix, classVector):
        """
        训练函数
        :param trainMatrix: 训练词向量矩阵
        :param classVector: 分类向量
        """

        numTrainNews = len(trainMatrix)
        # 多少条新闻
        numWords = len(trainMatrix[0])
        # 训练集一共多少个词语

        # pFood = sum(classVector) / float(numTrainNews)
        # 新闻属于食品安全类的概率
        pFood = float(1)/float(200)

        p0Num = ones(numWords)
        p1Num = ones(numWords)
        p0Sum = 2.0
        p1Sum = 2.0
        # 以上初始化概率，避免有零的存在使后面乘积结果为0
        # self.words_weight
        for i in range(numTrainNews):
            if classVector[i] == 1:
                # +=
                p1Num += trainMatrix[i]
                # 每条食品安全新闻中，每个词的数量分布
                p1Sum += sum(trainMatrix[i])
                # 求每条新闻出现的食品安全集合中的词语的数量总合
            else:
                # +=
                p0Num += trainMatrix[i]
                p0Sum += sum(trainMatrix[i])
        p1Vect = log(p1Num / p1Sum)
        # 在1的情况下每个词语出现的概率
        p0Vect = log(p0Num / p0Sum)
        # 在0的情况下每个词语出现的概率

        #保存结果
        self.p0V = p0Vect
        self.p1V = p1Vect
        self.p1 = pFood


    def classify_news(self, news):
        """
        分类函数,对输入新闻进行处理，然后分类
        :param vec2Classify: 欲分类的新闻
        """
        vectorized, vocabulary = vectorize_newslist([news],self.vocabulary)
        return self.classify_vector(vectorized)

    def classify_vector(self, vec2Classify):
        """
        分类函数,对输入词向量分类
        :param vec2Classify: 欲分类的词向量
        """
        vec2Classify = vec2Classify[0]
        p1 = sum(vec2Classify * self.p1V) + log(self.p1)
        # 元素相乘
        p0 = sum(vec2Classify * self.p0V) + log(1.0 - self.p1)
        if p1 - p0 > 0:
            return 1
        else:
            return 0

    def get_param(self):
        """
        获取模型参数，返回一个元组，元素分别是 反例概率向量 正例概率向量 正例先验概率
        """
        return (self.p0V,self.p1V,self.p1)


class scikitNB:
    """
    贝叶斯分类器, scikit版本
    保存词典，以供新闻处理
    """

    def __init__(self, vocabulary):
        self.model = MultinomialNB(alpha=0.1)
        # self.model = GaussianNB()
        self.vocabulary = vocabulary

    def train(self, trainMatrix, classVector):
        """
        训练函数
        :param trainMatrix: 训练词向量矩阵
        :param classVector: 分类向量
        """
        self.model.fit(trainMatrix,classVector)


    def classify_news(self, news):
        """
        分类函数,对输入新闻进行处理，然后分类
        :param news: 欲分类的新闻
        """
        vector = vectorize_newslist([news],self.vocabulary)[0]
        return self.classify_vector(vector[0])


    def classify_vector(self, vec2Classify):
        """
        FIXME:建议不要使用这个函数，统一使用上述
        分类函数,对输入词向量分类
        :param vec2Classify: 欲分类的词向量
        """
        #print vec2Classify
        return self.model.predict([vec2Classify])[0]


    def get_param(self):
        """
        获取模型参数，返回一个元组，元素分别是 反例概率向量 正例概率向量 正例先验概率
        """
        p0V = self.model.feature_log_prob_[0]
        p1V = self.model.feature_log_prob_[1]
        p1 = self.model.class_log_prior_[1]
        return (p0V, p1V, p1)

import numpy as np
import re
import os
import random
import linecache 
import jieba
import jieba.posseg as pseg
def jieba_cut_and_save_file(inputList, n_weight, a_weight, output_cleaned_file=False):
    """
    1. 读取中文文件并分词句子
    2. 可以将分词后的结果保存到文件
    3. 如果已经存在经过分词的数据文件则直接加载
    """
    output_file = os.path.join('./data/', 'cleaned_' + 'trainMatrix.txt')
    lines = []
    tags = []
    for line in inputList:
        result = pseg.cut(clean_str(line))
        a = []
        b = []
        for word, flag in result:
            # 对分词后的新闻
            if word != ' ':
                # 若非空
                a.append(word)
                if flag.find('n')!=0:
                    # 若是名词
                    b.append(n_weight)
                elif flag.find('a')!=0:
                    # 若形容词
                    b.append(a_weight)
                else:
                    b.append(1)
        lines.append(a)
        tags.append(b)
    if output_cleaned_file:
        with open(output_file, 'w') as f:
            for line in lines:
                f.write(" ".join(line) + '\n')

    vocabulary = createVocabList(lines)
    # 根据词典生成词向量化器,并进行词向量化
    setOfWords2Vec = setOfWords2VecFactory(vocabulary)
    vectorized = []
    for i,news in enumerate(lines):
        vector = setOfWords2Vec(news, tags[i])
        vectorized.append(vector)
    return vectorized, vocabulary

def loadTrainDataset(posFile,negFile):
    """
    便利函数，加载训练数据集

    :param pos: 多少条食品安全相关新闻
    :param neg: 多少条非食品安全相关新闻

    """

    trainingList = []  # 训练集
    classVec = []  # 分类向量

    # 录入食品安全相关的训练集
    posList = list(open(posFile, 'r').readlines())
    posVec = [1] * len(posList)
    trainingList += posList
    classVec += posVec

    # 录入非食品安全相关的训练集
    negList = list(open(negFile, 'r').readlines())
    negVec = [0] * len(negList)
    trainingList += negList
    classVec += negVec

    return trainingList, classVec

def clean_str(string):
    """
    1. 将除汉字外的字符转为一个空格
    2. 除去句子前后的空格字符
    """
    string = re.sub(r'[^\u4e00-\u9fff]', ' ', string)
    string = re.sub(r'\s{2,}', ' ', string)
    return string.strip()

def createVocabList(news_list):
    """
    从分词后的新闻列表中构造词典
    """
    # 创造一个包含所有新闻中出现的不重复词的列表。
    vocabSet = set([])
    for news in news_list:
        vocabSet = vocabSet | set(news)
        # |取并
    return list(vocabSet)

def vectorize_newslist(news_list, vocabulary):
    """
    将新闻列表新闻向量化，变成词向量矩阵
    注：如果没有词典，默认值为从集合中创造

    """
    # 分词与过滤
    cut_news_list = [list(jieba.cut(clean_str(news))) for news in news_list]

    # 根据词典生成词向量化器,并进行词向量化
    setOfWords2Vec = setOfWords2VecFactory(vocabulary)
    vectorized = [setOfWords2Vec(news) for news in cut_news_list]

    return vectorized, vocabulary

def setOfWords2VecFactory(vocabList):
    """
    通过给定词典，构造该词典对应的setOfWords2Vec
    """
    #优化：通过事先构造词语到索引的哈希表，加快转化
    index_map = {}
    for i, word in enumerate(vocabList):
        index_map[word] = i

    def setOfWords2Vec(news, tag=None):
        """
        以在构造时提供的词典为基准词典向量化一条新闻
        """
        result = [0]*len(vocabList)
        for i,word in enumerate(news):
                #通过默认值查询同时起到获得索引与判断有无的作用
                index = index_map.get(word, None)
                if index and tag == None:
                    result[index] = 1
                elif index and tag != None:
                    result[index] = tag[i]
        return result
    return setOfWords2Vec

def record_time_wrapper(stage_name,func):
    """
    包装函数,对函数进行计时
    """
    def timed(*args,**kwargs):
        start = time.clock()
        ret = func(*args, **kwargs)
        print (stage_name+"使用了：" + str(time.clock() - start) + '秒\n')
        return ret
    return timed

if __name__ == '__main__':
    print('')


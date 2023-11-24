import math
import operator
import pandas as pd


def loadDataSet():  # 读取存放样本数据的csv文件，返回样本数据集和划分属性集
    # User ID 的影响
    # dataSet = pd.read_csv("car_data.csv",  delimiter=',')
    dataSet = pd.read_csv("car_data.csv", usecols=[1, 2, 3, 4], nrows=40, delimiter=',')
    df = pd.DataFrame(dataSet)
    df['Income'] = (df['Income'] / 10000)
    df['Income'] = df['Income'].astype(int)
    labeSet = list(dataSet.columns)[:-1]  # 得到划分属性集
    df = df.values  # 得到样本数据集
    return df, labeSet


# 计算样本数据集的信息熵
def calcShannonEnt(dataSet):  # dataSet的每个元素是一个存放样本的属性值的列表
    numEntries = len(dataSet)  # 获取样本集个数
    labeCounts = {}  # 保存每个类别出现次数的字典
    for featVec in dataSet:  # 对每个样本进行统计
        currentLabel = featVec[-1]  # 取最后一列数据，即类别信息
        if currentLabel not in labeCounts.keys():
            labeCounts[currentLabel] = 0  # 添加字典元素，键值为0
        labeCounts[currentLabel] += 1  # 计算类别
    shannonEnt = 0.0  # 计算信息熵
    for key in labeCounts.keys():  # keys()以列表返回一个字典所有的键
        prob = float(labeCounts[key]) / numEntries  # 计算一个类别的概率
        shannonEnt -= prob * math.log(prob, 2)
    return shannonEnt


# 按照给定特征划分数据集
def splitDataSet(dataSet, axis, value):  # 返回按照axis属性切分，该属性的值为value的子数据集，并把该属性及其值去掉
    retDataSet = []
    for featVec in dataSet:  # dataSet的每个元素是一个样本，以列表表示，将相同特征值value的样本提取出来
        if featVec[axis] == value:  # 只把该属性上值是value的，加入到子数据集中
            reducedFeatVec = list(featVec[:axis])
            reducedFeatVec.extend(featVec[axis + 1:])  # extend()在列表list末尾一次性追加序列中的所有元素
            retDataSet.append(reducedFeatVec)
    return retDataSet  # 返回不含特征的子集


# 按照最大信息增益划分数据集
def chooseBestFeatureToSplit(dataSet):  # 选择出用于切分的标签的索引值
    numberFeatures = len(dataSet[0]) - 1  # 计算标签数量
    baseEntropy = calcShannonEnt(dataSet)  # 计算数据集的信息熵
    bestInfoGain = 0.0  # 初始增益0
    bestFeature = -1  # 最优划分属性的索引值
    for i in range(numberFeatures):  # 按信息增益选择标签索引
        featList = [example[i] for example in dataSet]  # 取出所有样本的第i个标签值
        uniqueVals = set(featList)  # 将标签值列表转为集合
        newEntropy = 0.0
        for value in uniqueVals:  # 对于一个属性i，对每个值切分成的子数据集计算其信息熵，将其加和就是总的熵
            subDataSet = splitDataSet(dataSet, i, value)  # 划分后的子集
            prob = len(subDataSet) / float(len(dataSet))  # 根据公式计算属性划分数据集的熵值
            newEntropy += prob * calcShannonEnt(subDataSet)  # 根据公式计算属性划分数据集的熵
        infoGain = baseEntropy - newEntropy  # 计算信息增益
        print("第%d个划分属性的信息增益为%.3f" % (i, infoGain))
        if (infoGain > bestInfoGain):  # 获取最大信息增益
            bestInfoGain = infoGain  # 更新信息增益，找到最大的信息增益
            bestFeature = i  # 记录信息增益最大的特征的索引值
    return bestFeature  # 返回信息增益最大特征的索引值


# 构建决策树
def majorityCnt(classList):  # 计算出现次数最多的标签，并返回该标签值
    classCount = {}
    for vote in classList:  # 投票法统计每个标签出现多少次
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    print(classCount.items())
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 对标签字典按值从大到小排序
    print(sortedClassCount)
    return sortedClassCount[0][0]  # 取已拍序的dic_items的第一个item的第一个值


def createTree(dataSet, labels):
    # 取出标签,生成列表
    classList = [example[-1] for example in dataSet]
    # 如果D中样本属于同一类别，则将node标记为该类叶节点
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果属性集为空或者D中样本在属性集上取值相同（意思是处理完所有特征后，标签还不唯一），
    # 将node标记为叶节点，类比为样本数量最多的类,这里采用字典做树结构，所以叶节点直接返回标签就行
    if len(dataSet[0]) == 1:  # 此时存在所有特征相同但标签不同的数据，需要取数量最多标签的作为叶子节点，数据集包含了标签，所以是1
        return majorityCnt(classList)  # 这种情况西瓜书中的属性集为空和样本集为空的情况

    bestFeat = chooseBestFeatureToSplit(dataSet)  # 选择最佳标签进行切分，返回标签索引
    bestFeatLabel = labels[bestFeat]  # 根据最佳标签索引取出该属性名
    myTree = {bestFeatLabel: {}}  # 定义嵌套的字典存放树结构
    del (labels[bestFeat])  # 属性名称列表中删除已选的属性名
    featVals = [example[bestFeat] for example in dataSet]  # 取出所有样本的最优属性的值
    uniqueVals = set(featVals)  # 将属性值转成集合，值唯一
    for value in uniqueVals:  # 对每个属性值继续递归切分剩下的样本
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


dataSet, labeSet = loadDataSet()
print(dataSet)
print('数据集的信息熵:', calcShannonEnt(dataSet))
print("最优索引值:" + str(chooseBestFeatureToSplit(dataSet)))
ID3_tree = createTree(dataSet, labeSet)
print('生成的决策树:', ID3_tree)

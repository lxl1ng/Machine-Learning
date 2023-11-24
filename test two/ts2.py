from pylab import *
import operator

# 特征字典
featureDic = {
    '色泽': ['浅白', '青绿', '乌黑'],
    '根蒂': ['硬挺', '蜷缩', '稍蜷'],
    '敲声': ['沉闷', '浊响', '清脆'],
    '纹理': ['清晰', '模糊', '稍糊'],
    '脐部': ['凹陷', '平坦', '稍凹'],
    '触感': ['硬滑', '软粘']}


def getDataSet():
    dataSet = [
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '好瓜'],
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '好瓜'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '好瓜'],
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜'],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '坏瓜'],
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '坏瓜'],
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '坏瓜'],
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '坏瓜'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜']
    ]

    features = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']

    # 每种特征的属性个数
    numList = []
    for i in range(len(features)):
        numList.append(len(featureDic[features[i]]))

    newDataSet = np.array(dataSet)
    # 得到训练数据集
    trainIndex = [0, 1, 2, 5, 6, 9, 13, 14, 15, 16, 3]
    trainDataSet = newDataSet[trainIndex]
    # 得到剪枝数据集
    pruneIndex = [4, 7, 8, 10, 11, 12]
    pruneDataSet = newDataSet[pruneIndex]

    return np.array(dataSet), trainDataSet, pruneDataSet, features


def calGini(dataArr):
    numEntries = dataArr.shape[0]
    classArr = dataArr[:, -1]
    uniqueClass = list(set(classArr))
    Gini = 1.0
    for c in uniqueClass:
        Gini -= (len(dataArr[dataArr[:, -1] == c]) / float(numEntries)) ** 2
    return Gini


def splitDataSet(dataSet, ax, value):
    """
    按照给点的属性ax和其中一种取值value来划分数据。
    """
    return np.delete(dataSet[dataSet[:, ax] == value], ax, axis=1)


def calSplitGin(dataSet, ax, labels):
    newGini = 0.0  # 划分完数据后的基尼指数
    # 对每一种属性
    for j in featureDic[ax]:
        axIndex = labels.index(ax)
        subDataSet = splitDataSet(dataSet, axIndex, j)
        prob = len(subDataSet) / float(len(dataSet))
        if prob != 0:  # prob为0意味着dataSet的ax属性中，没有第j+1种值
            newGini += prob * calGini(subDataSet)
    return newGini


def chooseBestSplit(dataSet, labelList):
    bestGain = 1
    bestFeature = -1
    n = dataSet.shape[1]
    # 对每一个特征
    for i in range(n - 1):
        newGini = calSplitGin(dataSet, labelList[i], labelList)
        if newGini < bestGain:
            bestFeature = i
            bestGain = newGini

    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount:
            classCount[vote] = 0
        classCount[vote] += 1
    # classCount.items()将字典的key-value对变成元组对，如{'a':1, 'b':2} -> [('a',1),('b',2)]
    # operator.itemgetter(1)按照第二个元素次序进行排序
    # reverse=True表示从大大到小。[('b',2), ('a',1)]
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1),
                              reverse=True)
    return sortedClassCount[0][0]  # 返回第0个元组的第0个值


def createTree(dataSet, labels):
    classList = dataSet[:, -1]
    # 如果基尼指数为0，即D中样本全属于同一类别，返回
    if calGini(dataSet) == 0:
        return dataSet[0][-1]
    # 属性值为空，只剩下类标签
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 得到增益最大划分的属性、值
    bestFeat = chooseBestSplit(dataSet, labels)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}  # 创建字典，即树的节点。
    # 生成子树的时候要将已遍历的属性删去。数值型不要删除。
    labelsCopy = labels[:]
    del (labelsCopy[bestFeat])
    uniqueVals = featureDic[bestFeatLabel]  # 最好的特征的类别列表
    for value in uniqueVals:  # 标称型的属性值有几种，就要几个子树。
        # Python中列表作为参数类型时，是按照引用传递的，要保证同一节点的子节点能有相同的参数。
        subLabels = labelsCopy[:]  # subLabels = 注意要用[:]，不然还是引用
        subDataSet = splitDataSet(dataSet, bestFeat, value)
        if len(subDataSet) != 0:
            myTree[bestFeatLabel][value] = createTree(subDataSet, subLabels)
        else:
            # 计算D中样本最多的类
            myTree[bestFeatLabel][value] = majorityCnt(classList)

    return myTree


def classify(data, featLabels, Tree):
    """
    通过决策树对一条数据分类
    """
    firstStr = list(Tree.keys())[0]  # 父节点
    secondDict = Tree[firstStr]  # 父节点下的子树，即子字典
    featIndex = featLabels.index(firstStr)  # 当前属性标识的位置
    classLabel = ""
    for key in secondDict.keys():  # 遍历该属性下的不同类
        if data[featIndex] == key:  # 如果数据中找到了匹配的属性类别
            # 如果不是叶子节点，继续向下遍历
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(data, featLabels, secondDict[key])
            # 如果是叶子节点，返回该叶子节点的类型
            else:
                classLabel = secondDict[key]
    return classLabel


def calAccuracy(dataSet, labels, Tree):
    """
    计算已有决策树的精度
    """
    cntCorrect = 0
    size = len(dataSet)
    for i in range(size):
        pre = classify(dataSet[i], labels, Tree)
        if pre == dataSet[i][-1]:
            cntCorrect += 1
    return cntCorrect / float(size)


def cntAccNums(dataSet, pruneSet):
    """
    用于剪枝，用dataSet中多数的类作为节点类，计算pruneSet中有多少类是被分类正确的，然后返回正确
    """
    nodeClass = majorityCnt(dataSet[:, -1])
    rightCnt = 0
    for vect in pruneSet:
        if vect[-1] == nodeClass:
            rightCnt += 1
    return rightCnt


# 预剪枝
def prePruning(dataSet, pruneSet, labels):
    classList = dataSet[:, -1]

    if calGini(dataSet) == 0:
        return dataSet[0][-1]

    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 获取最好特征
    bestFeat = chooseBestSplit(dataSet, labels)
    bestFeatLabel = labels[bestFeat]
    # 计算初始正确率
    baseRightNums = cntAccNums(dataSet, pruneSet)
    # 得到最好划分属性取值
    features = featureDic[bestFeatLabel]
    # 计算尝试划分节点时的正确率
    splitRightNums = 0.0
    for value in features:
        # 每个属性取值得到的子集
        subDataSet = splitDataSet(dataSet, bestFeat, value)
        if len(subDataSet) != 0:
            # 把用来剪枝的子集也按照相应属性值划分下去
            subPruneSet = splitDataSet(pruneSet, bestFeat, value)
            splitRightNums += cntAccNums(subDataSet, subPruneSet)
    if baseRightNums < splitRightNums:  # 如果不划分的正确点数少于尝试划分的点数，则继续划分。
        myTree = {bestFeatLabel: {}}
    else:
        return majorityCnt(dataSet[:, -1])  # 否则，返回不划分时投票得到的类

    # 以下代码和不预剪枝的代码大致相同，一点不同在于每次测试集也要参与划分。
    for value in features:
        subLabels = labels[:]
        subDataSet = splitDataSet(dataSet, bestFeat, value)
        subPruneSet = splitDataSet(pruneSet, bestFeat, value)
        if len(subDataSet) != 0:
            myTree[bestFeatLabel][value] = prePruning(subDataSet, subPruneSet, subLabels)
        else:
            # 计算D中样本最多的类
            myTree[bestFeatLabel][value] = majorityCnt(classList)
    return myTree


# 后剪枝
def postPruning(dataSet, pruneSet, labels):
    classList = dataSet[:, -1]
    # 如果基尼指数为0，即D中样本全属于同一类别，返回
    if calGini(dataSet) == 0:
        return dataSet[0][-1]
    # 属性值为空，只剩下类标签
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 得到增益最大划分的属性、值
    bestFeat = chooseBestSplit(dataSet, labels)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}  # 创建字典，即树的节点。
    # 生成子树的时候要将已遍历的属性删去。数值型不要删除。
    labelsCopy = labels[:]
    del (labelsCopy[bestFeat])
    uniqueVals = featureDic[bestFeatLabel]  # 最好的特征的类别列表
    for value in uniqueVals:  # 标称型的属性值有几种，就要几个子树。
        # Python中列表作为参数类型时，是按照引用传递的，要保证同一节点的子节点能有相同的参数。
        subLabels = labelsCopy[:]  # subLabels = 注意要用[:]，不然还是引用
        subPrune = splitDataSet(pruneSet, bestFeat, value)
        subDataSet = splitDataSet(dataSet, bestFeat, value)
        if len(subDataSet) != 0:
            myTree[bestFeatLabel][value] = postPruning(subDataSet, subPrune, subLabels)
        else:
            # 计算D中样本最多的类
            myTree[bestFeatLabel][value] = majorityCnt(classList)

    # 后剪枝，如果到达叶子节点，尝试剪枝。
    # 计算未剪枝时，测试集的正确数
    numNoPrune = 0.0
    for value in uniqueVals:
        subDataSet = splitDataSet(dataSet, bestFeat, value)
        if len(subDataSet) != 0:
            subPrune = splitDataSet(pruneSet, bestFeat, value)
            numNoPrune += cntAccNums(subDataSet, subPrune)
    # 计算剪枝后，测试集正确数
    numPrune = cntAccNums(dataSet, pruneSet)
    # 比较决定是否剪枝, 如果剪枝后该节点上测试集的正确数变多了，则剪枝。
    if numNoPrune < numPrune:
        return majorityCnt(dataSet[:, -1])  # 直接返回节点上训练数据的多数类为节点类。

    return myTree


dataSet, trainData, pruneData, labelList = getDataSet()
GiniTree = createTree(trainData, labelList)
print(GiniTree)
Gini_pretree = prePruning(trainData, pruneData, labelList)
print(Gini_pretree)
Gini_posttree = postPruning(trainData, pruneData, labelList)
print(Gini_posttree)
print(f"full tree's train accuracy = {calAccuracy(trainData, labelList, GiniTree)},"
      f"test accuracy = {calAccuracy(pruneData, labelList, GiniTree)}\n")
print(f"pre pruning tree's train accuracy = {calAccuracy(trainData, labelList, GiniTree)},"
      f"test accuracy = {calAccuracy(pruneData, labelList, Gini_pretree)}\n")
print(f"post pruning tree's train accuracy = {calAccuracy(trainData, labelList, GiniTree)},"
      f"test accuracy = {calAccuracy(pruneData, labelList, Gini_posttree)}\n")

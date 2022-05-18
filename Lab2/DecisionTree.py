#ID3 Decision Tree
import math
import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None):
        """ 初始化
        Arguments:
            max_depth: 最大深度 
        """
        self.max_depth = max_depth
    
    def Entropy(self, data, attribute):
        """ 计算熵
        Arguments:
            data: 数据集
            attribute: 特征列

        Returns:
            entropy: 熵
        """

        column, counts = np.unique(data[attribute], return_counts=True)
        entropy = 0.0
        for i in range(len(column)):
            entropy -= (counts[i] / np.sum(counts)) * math.log2(counts[i] / np.sum(counts))
        return entropy

    def InformationGain(self, data, feature, target):
        """ 计算信息增益
        Arguments:
            data: 数据集
            feature: 特征列名字
            target: 目标列名字
        
        Returns:
            information_gain: 信息增益
        """
        # 计算数据集的熵
        entropy = self.Entropy(data, target)
        # 计算目标特征列的熵
        column, counts = np.unique(data[feature], return_counts=True)
        entropy_feature = 0.0
        for i in range(len(column)):
            entropy_feature += (counts[i] / np.sum(counts)) * self.Entropy(data[data[feature] == column[i]], target)
        # 计算信息增益
        information_gain = entropy - entropy_feature
        return information_gain

    def ID3(self, data, raw, attributes, target, depth=None, parent=None):
        """ ID3决策树构建
        Arguments:
            data: 待划分的数据集
            raw: 原数据集
            attributes: 待划分的特征列
            target: 目标列名字
            depth: 当前深度
            parent: 父节点

        Returns:
            tree: 决策树
        """

        # 如果只有一类数据，直接返回此类
        if len(np.unique(data[target])) <= 1:
            return data[target].iloc[0]
        # 需要划分的数据集为空，说明已经没有可以划分的属性了，返回出现次数最多的类别
        if len(data) == 0:
            return np.unique(raw[target])[np.argmax(np.unique(raw[target], return_counts=True)[1])]
        # 如果已经到达了最大深度，返回出现次数最多的类别
        if self.max_depth is not None and depth >= self.max_depth:
            return np.unique(data[target])[np.argmax(np.unique(data[target], return_counts=True)[1])]
        # 获取信息增益最大的属性
        parent = np.unique(data[target])[np.argmax(np.unique(data[target], return_counts=True)[1])]
        # 计算信息增益
        information_gain = self.InformationGain(data, attributes[0], target)
        # 如果信息增益为0，返回出现次数最多的类别
        if information_gain == 0:
            return data[target].iloc[0]
        # 如果信息增益不为0，则划分数据集
        best_attribute = attributes[0]
        for attribute in attributes[1:]:
            if self.InformationGain(data, attribute, target) > information_gain:
                best_attribute = attribute
                information_gain = self.InformationGain(data, attribute, target)
        # 创建新的划分树
        tree = {best_attribute: {}}
        # 创建子树
        column = np.unique(data[best_attribute], return_counts=True)[0]
        # 在best_attribute属性上划分数据集
        for i in range(len(column)):
            # 创建子树
            tree[best_attribute][column[i]] = self.ID3(data.where(data[best_attribute] == column[i]).dropna(), raw, attributes.drop(best_attribute), target, depth + 1, parent)
        return tree

    def predict(self, data):
        """ 预测数据集
        Arguments:
            data: 数据集

        Returns:
            predictions: 预测结果
        """

        dict_data = data.to_dict(orient='records')
        result = []
        for i in range(len(dict_data)):
            result.append(self.predict_one(dict_data[i], self.tree))
        return result
    
    def predict_one(self, data, tree):
        """ 预测单个数据
        Arguments:
            data: 数据
            tree: 决策树

        Returns:
            prediction: 预测结果
        """

        for attribute in list(data.keys()):
            if attribute in list(tree.keys()):
                try:
                    result = tree[attribute][data[attribute]]
                except:
                    return 1
                result = tree[attribute][data[attribute]]
                if isinstance(result, dict):
                    return self.predict_one(data, result)
                else:
                    return result

    def fit(self, data, target):
        """ 训练决策树
        Arguments:
            data: 数据集
            target: 目标列名字
        """

        data1 = data.copy()
        data1[target.name] = target
        self.tree = self.ID3(data1, data1, data.columns, target.name, 0)
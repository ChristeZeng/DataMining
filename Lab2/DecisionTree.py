#ID3 Decision Tree
import math
import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
    
    def Entropy(self, data, attribute):
        column, counts = np.unique(data[attribute], return_counts=True)
        entropy = 0.0
        for i in range(len(column)):
            entropy -= (counts[i] / np.sum(counts)) * math.log2(counts[i] / np.sum(counts))
        return entropy

    def InformationGain(self, data, feature, target):
        # calcu entropy of target
        entropy = self.Entropy(data, target)
        # calcu entropy of feature
        column, counts = np.unique(data[feature], return_counts=True)
        entropy_feature = 0.0
        for i in range(len(column)):
            entropy_feature += (counts[i] / np.sum(counts)) * self.Entropy(data[data[feature] == column[i]], target)
        # calcu information gain
        information_gain = entropy - entropy_feature
        return information_gain

    def ID3(self, data, row, attributes, target, depth=None, parent=None):
        print("depth:", depth)
        print("class:", data[target].unique())
        # 如果只有一类数据，直接返回此类
        if len(np.unique(data[target])) <= 1:
            return data[target].iloc[0]
        # 需要划分的数据集为空，说明已经没有可以划分的属性了，返回出现次数最多的类别
        if len(data) == 0:
            return np.unique(row[target])[np.argmax(np.unique(row[target], return_counts=True)[1])]
        # # 如果已经到达了最大深度，返回出现次数最多的类别
        if self.max_depth is not None and depth >= self.max_depth:
            return np.unique(row[target])[np.argmax(np.unique(row[target], return_counts=True)[1])]
        # 如果没有划分属性，返回上次划分的结果 ??
        if len(attributes) == 0:
            return parent
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
        print("best_attribute:", best_attribute)
        # 创建新的划分树
        tree = {best_attribute: {}}
        # 创建子树
        column = np.unique(data[best_attribute], return_counts=True)[0]
        # 在best_attribute属性上划分数据集
        print(attributes.drop(best_attribute))
        for i in range(len(column)):
            # 创建子树
            tree[best_attribute][column[i]] = self.ID3(data.where(data[best_attribute] == column[i]).dropna(), row, attributes.drop(best_attribute), target, depth + 1, parent)
        # print("tree:", tree)
        return tree

    # def predict(self, data):
    #     dict_data = data.to_dict(orient='records')
    #     result = []
    #     for i in range(len(dict_data)):
    #         result.append(self.predict_one(dict_data[i], self.tree))
    #     return result
    
    def predict_one(self, data, tree):
        for key in list(tree.keys()):
            if key in list(data.keys()):
                if type(tree[key][data[key]]) is dict:
                    return self.predict_one(data, tree[key])
                else:
                    return tree[key]

    def fit(self, data, target):
        data1 = data.copy()
        data1[target.name] = target
        self.tree = self.ID3(data1, data1, data.columns, target.name, 0)

    def make_prediction(self, sample, tree, default=1):
        # map sample data to tree
        for attribute in list(sample.keys()):
            # check if feature exists in tree
            if attribute in list(tree.keys()):
                try:
                    result = tree[attribute][sample[attribute]]
                except:
                    return default

                result = tree[attribute][sample[attribute]]

                # if more attributes exist within result, recursively find best result
                if isinstance(result, dict):
                    return self.make_prediction(sample, result)
                else:
                    return result

    def predict(self, input):
        # convert input data into a dictionary of samples
        samples = input.to_dict(orient='records')
        predictions = []

        # make a prediction for every sample
        for sample in samples:
            predictions.append(self.make_prediction(sample, self.tree, 1.0))

        return predictions
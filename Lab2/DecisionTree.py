#ID3 Decision Tree
from itertools import count
import math
import numpy as np

class DecisionTree:
    def __init__(self, data, attributes, max_depth=None):
        self.data = data
        self.attributes = attributes
        self.max_depth = max_depth
    
    def Entropy(self, data, attributes):
        column, counts = np.unique(data[attributes], return_counts=True)
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

    def ID3(self, data, attributes, target, depth=0):
        # if all data is same, return the value
        if len(np.unique(data[target])) <= 1:
            return data[target].iloc[0]
        # if all attributes are used, return the value
        if len(attributes) == 0:
            return data[target].iloc[0]
        # if max_depth is reached, return the value
        if self.max_depth is not None and depth >= self.max_depth:
            return data[target].iloc[0]
        # calcu information gain
        information_gain = self.InformationGain(data, attributes[0], target)
        # if information gain is 0, return the value
        if information_gain == 0:
            return data[target].iloc[0]
        # calcu best attribute
        best_attribute = attributes[0]
        for attribute in attributes[1:]:
            if self.InformationGain(data, attribute, target) > information_gain:
                best_attribute = attribute
        # create a new decision tree
        tree = {best_attribute: {}}
        # get all unique values of best_attribute
        column, counts = np.unique(data[best_attribute], return_counts=True)
        # for each unique value of best_attribute
        for i in range(len(column)):
            # create a new decision tree
            tree[best_attribute][column[i]] = self.ID3(data[data[best_attribute] == column[i]], attributes[:].drop(best_attribute), target, depth + 1)
        return tree

    def predict(self, tree, data):
        # if the tree is a value, return the value
        if not isinstance(tree, dict):
            return tree
        # get the best attribute
        best_attribute = list(tree.keys())[0]
        # get the best attribute value
        best_attribute_value = data[best_attribute]
        # get the next tree
        next_tree = tree[best_attribute][best_attribute_value]
        # return the prediction
        return self.predict(next_tree, data)

    def fit(self):
        self.tree = self.ID3(self.data, self.attributes, self.data.target)
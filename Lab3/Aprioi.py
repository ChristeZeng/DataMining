from itertools import chain, combinations
from webbrowser import get


class Aprioi:
    
    def __init__(self, ItemSetList, MinSupport, MinConfidence):
        """
        构造函数
        :param ItemSetList: 数据集
        :param MinSupport: 最小支持度
        :param MinConfidence: 最小置信度
        """

        self.ItemSetList = ItemSetList
        self.MinSupport = MinSupport
        self.MinConfidence = MinConfidence

    def apriori(self):
        """
        Apriori算法
        :return: 返回频繁集
        :return: 返回关联规则
        """
        frequentItems = {}
        frequentItemsAll = {}

        # 将嵌套列表转为嵌套集合（集合使用hashtable的方式进行组织）
        C1ItemSet = set()
        for itemSet in self.ItemSetList:
            for item in itemSet:
                C1ItemSet.add(frozenset([item]))
        # 算法的第一步骤，获取频繁1-项集
        L1ItemsSet = self.getFrequentItems(C1ItemSet, frequentItemsAll)
        # 从k = 2开始循环判断
        CurLSet = L1ItemsSet
        k = 2
        while CurLSet:
            frequentItems[k - 1] = CurLSet
            # 通过连接操作获取候选频繁k-项集
            CkItemSet = set([i.union(j) for i in CurLSet for j in CurLSet if len(i.union(j)) == k])
            # 删除不满足规则的组合
            temp = CkItemSet.copy()
            for item in CkItemSet:
                subsets = combinations(item, k - 1)
                for subset in subsets:
                    if(frozenset(subset) not in CurLSet):
                        temp.remove(item)
                        break
            CkItemSet = temp
            # 算法的第k步骤，获取频繁k-项集
            CurLSet = self.getFrequentItems(CkItemSet, frequentItemsAll)
            # 继续迭代
            k += 1

        # 通过频繁项集获取关联规则
        associationRules = self.getAssociationRules(frequentItems, frequentItemsAll)
        return frequentItems, associationRules
    
    
    def getFrequentItems(self, ItemSet, frequentItemsAll):
        """
        获取频繁集
        :param ItemSet: 集合
        :param frequentItemsAll: 频繁集
        :return: 返回频繁集
        """
        frequentItemsSet = set()
        frequentItems = {}
        for item in ItemSet:
            for itemSet in self.ItemSetList:
                if item.issubset(itemSet):
                    if item in frequentItems:
                        frequentItems[item] += 1
                    else:
                        frequentItems[item] = 1
                    
                    if item in frequentItemsAll:
                        frequentItemsAll[item] += 1
                    else:
                        frequentItemsAll[item] = 1
        
        for item, sup in frequentItems.items():
            if float(sup / len(self.ItemSetList)) >= self.MinSupport:
                frequentItemsSet.add(item)
        
        return frequentItemsSet

    def getAssociationRules(self, frequentItems, frequentItemsAll):
        """
        获取关联规则
        :param frequentItems: 频繁集
        :param frequentItemsAll: 含有支持度的频繁集
        :return: 返回关联规则
        """
        associationRules = []
        for num, itemSet in frequentItems.items():
            for item in itemSet:
                # 计算item的powerset
                subsets = chain.from_iterable(combinations(item, r) for r in range(1, len(item)))
                # 对于powerset中的每一个子集，计算置信度
                for s in subsets:
                    # 通过置信度计算关联规则
                    confidence = float(frequentItemsAll[item] / frequentItemsAll[frozenset(s)])
                    if(confidence >= self.MinConfidence):
                        associationRules.append([set(s), set(item.difference(s)), confidence])
        
        return associationRules
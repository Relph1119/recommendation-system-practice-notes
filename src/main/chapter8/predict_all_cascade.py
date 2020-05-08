#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: predict_all.py
@time: 2020/5/8 17:12
@project: recommendation-system-practice-notes
@desc: 用户分类对物品分类的平均值算法
"""


class PredictAllCascade:
    def __init__(self, train_dataset, test_dataset, user_group, item_group):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.group = {}
        self.user_cluster = user_group(train_dataset)
        self.item_cluster = item_group(train_dataset)

    def _train(self, group, records):
        for r in records:
            ug = self.user_cluster.get_group(r.user)
            ig = self.item_cluster.get_group(r.item)
            if ug not in group:
                group[ug] = {}
            if ig not in group[ug]:
                group[ug][ig] = []
            # 这里计算的残差
            group[ug][ig].append(r.rate - r.predict)

    def fit(self):
        group = {}
        self._train(group, self.train_dataset)
        self._train(group, self.test_dataset)
        for ug in group:
            for ig in group[ug]:
                group[ug][ig] = sum(group[ug][ig]) / (1.0 * len(group[ug][ig]) + 1.0)
        self.group = group

        return self.predict(self.train_dataset)

    def predict(self, records):
        for r in records:
            ug = self.user_cluster.get_group(r.user)
            ig = self.item_cluster.get_group(r.item)
            r.predict += self.group[ug][ig]

        return records

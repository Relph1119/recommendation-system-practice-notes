#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: demographic_most_popular.py
@time: 2020/5/9 1:26
@project: recommendation-system-practice-notes
@desc: DemographicMostPopular算法
"""
from main.chapter3.most_popular import MostPopular


class DemographicMostPopular:
    def __init__(self, train_dataset, profile):
        # 训练数据
        self.train_dataset = train_dataset
        # 用户的注册信息
        self.profile = profile
        self.model = None
        self.items = {}

    def fit(self):
        # 建立多重字典，将缺失值当成other，同归为一类进行处理
        items = {}
        for user in self.train_dataset.keys():
            gender = self.profile[user]['gender']
            if gender not in items:
                items[gender] = {}
            age = self.profile[user]['age'] // 10
            if age not in items[gender]:
                items[gender][age] = {}
            country = self.profile[user]['country']
            if country not in items[gender][age]:
                items[gender][age][country] = {}
            for item in self.train_dataset[user]:
                if item not in items[gender][age][country]:
                    items[gender][age][country][item] = 0
                items[gender][age][country][item] += 1
        for gender in items:
            for age in items[gender]:
                for country in items[gender][age]:
                    items[gender][age][country] = list(
                        sorted(items[gender][age][country].items(), key=lambda x: x[1], reverse=True))

        self.items = items

    def recommend_users(self, users, N):
        """
        给用户推荐的商品
        :param users: 用户
        :param N: 超参数，设置取TopN推荐物品数目
        :return: 推荐字典 {用户 : 推荐的商品list}
        """
        recommend_items = dict()
        for user in users:
            user_recommends = self.recommend(user, N)
            recommend_items[user] = user_recommends

        return recommend_items

    def recommend(self, user, N):
        seen_items = set(self.train_dataset[user]) if user in self.train_dataset else set()
        gender = self.profile[user]['gender']
        age = self.profile[user]['age']
        country = self.profile[user]['country']
        if gender not in self.items.keys() or age not in self.items[gender] or country not in self.items[gender][age]:
            if self.model is None:
                self.model = MostPopular(self.train_dataset, self.profile)
                self.model.fit()
            recs = self.model.recommend(user, N)
        else:
            recs = [x[0] for x in self.items[gender][age][country] if x[0] not in seen_items][:N]
        return recs

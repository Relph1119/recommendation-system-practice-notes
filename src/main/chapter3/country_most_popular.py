#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: country_most_popular.py
@time: 2020/5/9 1:21
@project: recommendation-system-practice-notes
@desc: CountryMostPopular算法
"""
from main.chapter3.most_popular import MostPopular


class CountryMostPopular:
    def __init__(self, train_dataset, profile):
        # 训练数据
        self.train_dataset = train_dataset
        # 用户的注册信息
        self.profile = profile
        self.model = None
        self.items = {}

    def fit(self):
        # 分城市进行统计
        items = {}
        for user in self.train_dataset.keys():
            country = self.profile[user]['country']
            if country not in items:
                items[country] = {}
            for item in self.train_dataset[user]:
                if item not in items[country]:
                    items[country][item] = 0
                items[country][item] += 1
        for country in items:
            items[country] = list(sorted(items[country].items(), key=lambda x: x[1], reverse=True))
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
        country = self.profile[user]['country']
        if country in self.items.keys():
            recs = [x[0] for x in self.items[country] if x[0] not in seen_items][:N]
        else:  # 没有提供城市信息的，按照全局推荐
            if self.model is None:
                self.model = MostPopular(self.train_dataset, self.profile)
                self.model.fit()
            recs = self.model.recommend(user, N)
        return recs

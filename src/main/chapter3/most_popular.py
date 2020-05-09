#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: most_popular.py
@time: 2020/5/8 23:27
@project: recommendation-system-practice-notes
@desc: MostPopular算法
"""


class MostPopular:
    def __init__(self, train_dataset, profile):
        # 训练数据
        self.train_dataset = train_dataset
        # 用户的注册信息
        self.profile = profile
        self.items = {}

    def fit(self):
        items = {}
        for user in self.train_dataset:
            for item in self.train_dataset[user]:
                if item not in items.keys():
                    items[item] = 0
                items[item] += 1
        self.items = list(sorted(items.items(), key=lambda x: x[1], reverse=True))

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
        seen_items = set(self.train_dataset[user]) if user in self.train_dataset.keys() else set()
        recs = [x[0] for x in self.items if x[0] not in seen_items][:N]
        return recs

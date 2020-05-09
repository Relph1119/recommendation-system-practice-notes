#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: gender_most_popular.py
@time: 2020/5/9 0:35
@project: recommendation-system-practice-notes
@desc: GenderMostPopular算法
"""
from main.chapter3.most_popular import MostPopular


class GenderMostPopular:
    def __init__(self, train_dataset, profile):
        # 训练数据
        self.train_dataset = train_dataset
        # 用户的注册信息
        self.profile = profile
        self.m_items, self.f_items = {}, {}
        self.model = None

    def fit(self):
        mitems, fitems = {}, {}  # 男、女
        tmp = {}
        for user in self.train_dataset:
            if self.profile[user]['gender'] == 'm':
                tmp = mitems
            elif self.profile[user]['gender'] == 'f':
                tmp = fitems
            for item in self.train_dataset[user]:
                if item not in tmp.keys():
                    tmp[item] = 0
                tmp[item] += 1
        self.m_items = list(sorted(mitems.items(), key=lambda x: x[1], reverse=True))
        self.f_items = list(sorted(fitems.items(), key=lambda x: x[1], reverse=True))

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
        if self.profile[user]['gender'] == 'm':
            recs = [x[0] for x in self.m_items if x[0] not in seen_items][:N]
        elif self.profile[user]['gender'] == 'f':
            recs = [x[0] for x in self.f_items if x[0] not in seen_items][:N]
        else:  # 没有提供性别信息的，按照MostPopular推荐
            if self.model is None:
                self.model = MostPopular(self.train_dataset, self.profile)
                self.model.fit()
            recs = self.model.recommend(user, N)
        return recs

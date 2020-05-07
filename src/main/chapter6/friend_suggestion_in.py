#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: friend_suggestion_in.py
@time: 2020/5/7 19:40
@project: recommendation-system-practice-notes
@desc: 基于社交网络的入度推荐算法
"""
import math


class FriendSuggestionIn:
    def __init__(self, dataset):
        self.G, self.GT = dataset

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
        """
        利用用户入度计算相似度
        :param user: 用户
        :param N: 超参数，设置取TopN推荐物品数目
        :return: 推荐的商品list
        """
        if user not in self.GT.keys():
            return []
        # 根据相似度推荐N个未见过的
        user_sim = {}
        user_friends = set(self.GT[user])
        for u in self.GT[user]:
            if u not in self.G.keys():
                continue
            for v in self.G[u]:
                if v != user and v not in user_friends:
                    if v not in user_sim:
                        user_sim[v] = 0
                    user_sim[v] += 1
        user_sim = {v: user_sim[v] / math.sqrt(len(self.GT[user]) * len(self.GT[v])) for v in user_sim}
        recs = list(sorted(user_sim.items(), key=lambda x: x[1], reverse=True))[:N]

        return [x[0] for x in recs]

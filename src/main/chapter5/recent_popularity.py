#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: recent_popularity.py
@time: 2020/4/27 19:54
@project: recommendation-system-practice-notes
@desc: 最近最热门推荐算法
"""
import time


class RecentPopular:
    def __init__(self, dataset, alpha=1.0, t0=int(time.time())):
        """
        :param dataset: 训练数据集
        :param alpha: 时间衰减因子
        :param t0: 当前的时间戳
        """
        self.item_score = {}
        self.dataset = dataset
        self.alpha = alpha
        self.t0 = t0

    def fit(self):
        for user in self.dataset:
            for item, timestamp in self.dataset[user]:
                if item not in self.item_score:
                    self.item_score[item] = 0
                self.item_score[item] += 1.0 / (self.alpha * (self.t0 - timestamp))

        self.item_score = sorted(self.item_score.items(), key=lambda x: x[1], reverse=True)

    def recommend_users(self, users, N, K=None):
        """
        给用户推荐的商品
        :param users: 用户
        :param K: 不需要，为了统一调用
        :param N: 超参数，设置取TopN推荐物品数目
        :return: 推荐字典 {用户 : 推荐的商品的list}
        """
        recommend_items = dict()
        for user in users:
            user_recommends = self.recommend(user, N)
            recommend_items[user] = user_recommends

        return recommend_items

    def recommend(self, user, N):
        # 随机推荐N个未见过的
        user_items = set()
        for item, _ in self.dataset[user]:
            user_items.add(item)
        rec_items = [x[0] for x in self.item_score if x[0] not in user_items]
        return rec_items[:N]

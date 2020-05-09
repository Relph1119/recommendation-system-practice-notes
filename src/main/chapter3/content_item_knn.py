#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: content_item_knn.py
@time: 2020/5/9 11:27
@project: recommendation-system-practice-notes
@desc: ContentItemKNN算法
"""
import math


class ContentItemKNN:
    def __init__(self, train_dataset, content):
        self.dataset = train_dataset
        self.content = content
        self.sorted_item_sim = {}
        self.item_sim = {}

    def fit(self):
        # 建立word-item倒排表
        word_item = {}
        for item in self.content.keys():
            for word in self.content[item]:
                if word not in word_item:
                    word_item[word] = {}
                word_item[word][item] = 1

        for word in word_item:
            for item in word_item[word]:
                word_item[word][item] /= math.log(1 + len(word_item[word]))

        # 计算相似度
        item_sim = {}
        mo = {}
        for word in word_item:
            for u in word_item[word]:
                if u not in item_sim:
                    item_sim[u] = {}
                    mo[u] = 0
                mo[u] += word_item[word][u] ** 2
                for v in word_item[word]:
                    if u == v: continue
                    if v not in item_sim[u]:
                        item_sim[u][v] = 0
                    item_sim[u][v] += word_item[word][u] * word_item[word][v]
        for u in item_sim:
            for v in item_sim[u]:
                item_sim[u][v] /= math.sqrt(mo[u] * mo[v])

        self.item_sim = item_sim
        # 按照相似度排序
        sorted_item_sim = {k: list(sorted(v.items(), key=lambda x: x[1], reverse=True))
                           for k, v in item_sim.items()}
        self.sorted_item_sim = sorted_item_sim

    def recommend_users(self, users, N, K):
        """
        给用户推荐的商品
        :param users: 用户
        :param N: 超参数，设置取TopN推荐物品数目
        :return: 推荐字典 {用户 : 推荐的商品list}
        """
        recommend_items = dict()
        for user in users:
            user_recommends = self.recommend(user, N, K)
            recommend_items[user] = user_recommends

        return recommend_items

    def recommend(self, user, N, K):
        items = {}
        seen_items = set(self.dataset[user])
        for item in self.dataset[user]:
            for u, _ in self.sorted_item_sim[item][:K]:
                # 删除用户已经见过的物品
                if u not in seen_items:
                    if u not in items:
                        items[u] = 0
                    items[u] += self.item_sim[item][u]
        recs = list(sorted(items.items(), key=lambda x: x[1], reverse=True))[:N]
        recs = [x[0] for x in recs]
        return recs

#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: age_most_popular.py
@time: 2020/5/9 1:01
@project: recommendation-system-practice-notes
@desc: AgeMostPopular算法
"""
from main.chapter3.most_popular import MostPopular


class AgeMostPopular:
    def __init__(self, train_dataset, profile):
        # 训练数据
        self.train_dataset = train_dataset
        # 用户的注册信息
        self.profile = profile
        self.age_items = []
        self.model = None

    def fit(self):
        # 对年龄进行分段
        ages = []
        for user in self.profile:
            if self.profile[user]['age'] >= 0:
                ages.append(self.profile[user]['age'])
        max_age, min_age = max(ages), min(ages)
        items = [{} for _ in range(int(max_age // 10 + 1))]
        # 分年龄段进行统计
        for user in self.train_dataset.keys():
            if self.profile[user]['age'] >= 0:
                age = self.profile[user]['age'] // 10
                for item in self.train_dataset[user]:
                    if item not in items[age]:
                        items[age][item] = 0
                    items[age][item] += 1
        for i in range(len(items)):
            items[i] = list(sorted(items[i].items(), key=lambda x: x[1], reverse=True))

        self.age_items = items

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
        if self.profile[user]['age'] >= 0:
            age = self.profile[user]['age'] // 10
            # 年龄信息异常的，按照全局推荐
            if age >= len(self.age_items) or len(self.age_items[age]) == 0:
                self._check_most_popular_model()
                recs = self.model.recommend(user, N)
            else:
                recs = [x for x in self.age_items[age] if x[0] not in seen_items][:N]
        else:  # 没有提供年龄信息的，按照全局推荐
            self._check_most_popular_model()
            recs = self.model.recommend(user, N)
        return recs

    def _check_most_popular_model(self):
        if self.model is None:
            self.model = MostPopular(self.train_dataset, self.profile)
            self.model.fit()

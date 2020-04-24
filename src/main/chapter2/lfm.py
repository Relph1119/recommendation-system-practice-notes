#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: main.py
@time: 2020/4/20 19:14
@project: recommendation-system-practice-notes
@desc: LFM隐语义模型
"""

import random
import sys
from operator import itemgetter

import numpy as np

from main.chapter2.usercf import UserCF
from main.util.utils import load_file, save_file

filepath = "store/LFM"


class LFM(UserCF):
    """隐语义模型"""

    def __init__(self, origin_data, F, ratio, learning_rate=0.02, regularization_rate=0.01):
        """
        :param origin_data: 原始数据集
        :param F: 隐特征的个数
        :param ratio: 负样本/正样本比例
        :param learning_rate: 梯度下降的学习率
        :param regularization_rate: 正则化率
        """
        self.ratio = ratio
        self.regularization_rate = regularization_rate
        self.learning_rate = learning_rate
        self.F = F

        self.user_p = None
        self.movie_q = None
        super().__init__(origin_data)

        # 初始化训练集
        self._init_train()
        # 获取商品集合
        self.item_pool = self.__get_items()

    def __get_items(self):
        """返回所有商品集"""
        print("开始计算商品集.....", file=sys.stderr)
        items_pool = set()
        for user, items in self.train_dataset.items():
            items_pool = items_pool.union(items)
        print("商品集计算完成", file=sys.stderr)
        return list(items_pool)

    def __build_matrix(self):
        """建立隐语义矩阵"""
        print("开始建立隐语义矩阵....", file=sys.stderr)
        # 建立用户的隐语义
        user_p = dict()
        for user in self.train_dataset.keys():
            user_p[user] = np.random.normal(size=self.F)

        movie_q = dict()
        for movie in self.item_pool:
            movie_q[movie] = np.random.normal(size=self.F)
        print("隐语义矩阵建立完成", file=sys.stderr)
        return user_p, movie_q

    def select_negatives(self, user_movies):
        """
        采集负样本,使负样本数量和正样本相同
        :param user_movies: 用户对应的正样本
        :return: 正负样本
        """
        items = dict()
        # 采集正样本
        for movie in user_movies:
            items[movie] = 1
        # print("开始采集负样本...")
        # 采集负样本
        n_negative = 0
        negtive_num = int(round(len(user_movies) * self.ratio))

        while n_negative < negtive_num and n_negative + len(user_movies) < len(self.item_pool):
            negitive_sample = random.choice(self.item_pool)
            if negitive_sample in items:
                continue
            items[negitive_sample] = 0
            n_negative += 1
        # print("完成采集负样本...")
        return items

    def train(self, lfm_matrix_path="store/lfm.pkl", epochs=1):
        print("开始训练模型")
        self.__get_lfm_matrix(lfm_matrix_path)

        for epoch in range(epochs):
            print("开始第{}轮训练".format(epoch + 1))
            for user, user_movies in self.train_dataset.items():
                select_samples = self.select_negatives(user_movies)
                for movie, rui in select_samples.items():
                    eui = rui - self.predict(user, movie)
                    user_latent = self.user_p[user]
                    movie_latent = self.movie_q[movie]

                    self.user_p[user] += self.learning_rate * (
                            movie_latent * eui - self.regularization_rate * user_latent)
                    self.movie_q[movie] += self.learning_rate * (
                            user_latent * eui - self.regularization_rate * movie_latent)
            self.learning_rate *= 0.9
            print("第{}轮完成".format(epoch + 1))

    def __get_lfm_matrix(self, lfm_matrix_path):
        try:
            print("开始载入隐语义矩阵....")
            self.user_p, self.movie_q = load_file(lfm_matrix_path)
            print("载入隐语义矩阵完成")
        except FileNotFoundError:
            print("载入隐语义矩阵失败，重新计算隐语义矩阵")
            # 建立隐语义矩阵
            self.user_p, self.movie_q = self.__build_matrix()
        print("开始保存隐语义矩阵")
        save_file(lfm_matrix_path, (self.user_p, self.movie_q))
        print("保存隐语义矩阵完成")

    def predict(self, user, item):
        return self.sigmoid(np.dot(self.user_p[user], self.movie_q[item]))

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    def recommend(self, user, N, K):
        """
        推荐
        :param user: 用户
        :param N: 推荐的商品个数
        :param K: 查找最相似的用户个数
        :return: 商品字典 {商品 : 相似性打分情况}
        """
        related_items = self.train_dataset.get(user, set)
        recommmend_items = dict()
        rank = dict()
        for f, puf in enumerate(self.user_p[user]):
            for movie, qfi in self.movie_q.items():
                if movie not in rank:
                    rank.setdefault(movie, 0.)
                    rank[movie] += puf * qfi[f]

        for movie, weight in sorted(rank.items(), key=itemgetter(1), reverse=True):
            if movie in related_items:
                continue
            recommmend_items.setdefault(movie, 0.)
            recommmend_items[movie] += weight

        return dict(sorted(recommmend_items.items(), key=itemgetter(1), reverse=True)[: N])

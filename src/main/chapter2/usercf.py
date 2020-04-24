#! /usr/bin/python3
# coding=utf-8
"""
Created on 2018年6月12日
@author: qcymkxyc
@desc: UserCF算法
"""
import math
import sys
from collections import defaultdict
from operator import itemgetter

from main.util.utils import load_file, save_file


class UserCF:
    """用户协同过滤"""

    def __init__(self, origin_data):
        self.origin_data = origin_data
        self.user_sim_matrix = None

    def train(self, sim_matrix_path="store/user_sim.pkl"):
        """
        训练模型
        :param sim_matrix_path: 协同矩阵保存的路径
        :return:
        """
        # 初始化训练集
        self._init_train()
        print("开始训练模型", file=sys.stderr)
        try:
            print("开始载入用户协同矩阵....", file=sys.stderr)
            self.user_sim_matrix = load_file(sim_matrix_path)
            print("载入协同过滤矩阵完成", file=sys.stderr)
        except FileNotFoundError:
            print("载入用户协同过滤矩阵失败，重新计算协同过滤矩阵", file=sys.stderr)
            # 计算用户协同矩阵
            self.user_sim_matrix = self.user_similarity()

        print("开始保存协同过滤矩阵", file=sys.stderr)
        save_file(sim_matrix_path, self.user_sim_matrix)
        print("保存协同过滤矩阵完成", file=sys.stderr)

    def _init_train(self):
        """
        初始化训练集数据
        :return:
        """
        self.train_dataset = dict()
        for user, item, _ in self.origin_data:
            self.train_dataset.setdefault(user, set())
            self.train_dataset[user].add(item)

    def user_similarity(self):
        """
        建立用户的协同过滤矩阵
        :return:
        """
        # 建立用户倒排表
        item_user = dict()
        for user, items in self.train_dataset.items():
            for item in items:
                item_user.setdefault(item, set())
                item_user[item].add(user)

        # 建立用户协同过滤矩阵
        user_sim_matrix = dict()
        N = defaultdict(int)  # 记录用户购买商品数
        for item, users in item_user.items():
            for u in users:
                N[u] += 1
                for v in users:
                    if u == v:
                        continue
                    user_sim_matrix.setdefault(u, defaultdict(int))
                    user_sim_matrix[u][v] += 1

        # 计算相关度
        for u, related_users in user_sim_matrix.items():
            for v, con_items_count in related_users.items():
                user_sim_matrix[u][v] = con_items_count / math.sqrt(N[u] * N[v])

        return user_sim_matrix

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
        for v, sim in sorted(self.user_sim_matrix.get(user, dict).items(),
                             key=itemgetter(1), reverse=True)[:K]:
            for item in self.train_dataset[v]:
                if item in related_items:
                    continue
                recommmend_items.setdefault(item, 0.)
                recommmend_items[item] += sim

        return dict(sorted(recommmend_items.items(), key=itemgetter(1), reverse=True)[: N])

    def recommend_users(self, users, N, K):
        """
        给用户推荐的商品
        :param users: 用户list
        :param N: 推荐的商品个数
        :param K: 查找最相似的用户个数
        :return: 推荐字典 {用户 : 推荐的商品的list}
        """
        recommend_items = dict()
        for user in users:
            user_recommends = list(self.recommend(user, N, K).keys())
            recommend_items[user] = user_recommends

        return recommend_items

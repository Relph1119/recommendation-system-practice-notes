#!/usr/bin/python3
# coding=utf-8
"""
Created on 2018年6月24日
@author: qcymkxyc
@Desc: ItemCF-IUF算法
"""
import math
from collections import defaultdict

from main.chapter2.itemcf import ItemCF


class ItemIUF(ItemCF):
    """
    ItemCF-IUF推荐算法
    """

    def train(self, sim_matrix_path="store/itemiuf_sim.pkl"):
        ItemCF.train(self, sim_matrix_path=sim_matrix_path)

    def _item_similarity(self):
        # 物品的协同矩阵
        item_sim_matrix = dict()
        # 每个物品的流行度
        N = defaultdict(int)

        # 统计同时购买商品的人数
        for _, items in self.train_dataset.items():
            for i in items:
                item_sim_matrix.setdefault(i, dict())
                # 统计商品的流行度
                N[i] += 1

                for j in items:
                    if i == j:
                        continue
                    item_sim_matrix[i].setdefault(j, 0)
                    item_sim_matrix[i][j] += 1. / math.log1p(len(items) * 1.)

        # 计算物品协同矩阵
        for i, related_items in item_sim_matrix.items():
            for j, related_count in related_items.items():
                item_sim_matrix[i][j] = related_count / math.sqrt(N[i] * N[j])

        return item_sim_matrix

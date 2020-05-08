#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: item_cluster.py
@time: 2020/5/8 10:51
@project: recommendation-system-practice-notes
@desc:
"""

from main.chapter8.cluster import Cluster


class ItemPopularityCluster(Cluster):

    def __init__(self, records):
        Cluster.__init__(self, records)
        popularity = {}
        for r in records:
            if r.item not in popularity:
                popularity[r.item] = 0
            popularity[r.item] += 1
        # 按照物品流行度进行分组
        k = 0
        for item, n in sorted(popularity.items(), key=lambda x: x[-1], reverse=False):
            c = int((k * 5) / len(popularity))
            self.group[item] = c
            k += 1

    def get_group(self, item):
        if item not in self.group:
            return -1
        else:
            return self.group[item]


class ItemVoteCluster(Cluster):

    def __init__(self, records):
        Cluster.__init__(self, records)
        vote, count = {}, {}
        for r in records:
            if r.item not in vote:
                vote[r.item] = 0
                count[r.item] = 0
            vote[r.item] += r.rate
            count[r.item] += 1
        # 按照物品平均评分进行分组
        for item, v in vote.items():
            c = v / (count[item] * 1.0)
            self.group[item] = int(c * 2)

    def get_group(self, item):
        if item not in self.group:
            return -1
        else:
            return self.group[item]

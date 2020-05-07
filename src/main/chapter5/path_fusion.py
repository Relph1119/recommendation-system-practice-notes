#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: path_fusion.py
@time: 2020/5/7 11:35
@project: recommendation-system-practice-notes
@desc: 时间段图模型
"""
import time


class PathFusion:
    def __init__(self, G, alpha, t=time.time()):
        self.G = G
        self.alpha = alpha
        self.time = str(t)

    def recommend(self, user):
        Q = []
        V = set()
        depth = dict()
        rank = dict()
        depth['u:' + user] = 0
        depth['ut:' + user + '_' + self.time] = 0
        rank['u:' + user] = self.alpha
        rank['ut:' + user + '_' + self.time] = 1 - self.alpha
        Q.append('u:' + user)
        Q.append('ut:' + user + '_' + self.time)
        while len(Q) > 0:
            v = Q.pop()
            if v in V:
                continue
            if depth[v] > 3:
                continue
            for v2, w in self.G[v].items():
                if v2 not in V:
                    depth[v2] = depth[v] + 1
                    Q.append(v2)
                rank[v2] = rank[v] * w

        return rank

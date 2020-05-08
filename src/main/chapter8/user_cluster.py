#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: user_cluster.py
@time: 2020/5/8 11:04
@project: recommendation-system-practice-notes
@desc:
"""
from main.chapter8.cluster import Cluster


class UserActivityCluster(Cluster):

    def __init__(self, records):
        Cluster.__init__(self, records)
        activity = {}
        for r in records:
            if r.user not in activity:
                activity[r.user] = 0
            activity[r.user] += 1
        # 按照用户活跃度进行分组
        k = 0
        for user, n in sorted(activity.items(), key=lambda x: x[-1], reverse=False):
            c = int((k * 5) / len(activity))
            self.group[user] = c
            k += 1

    def get_group(self, uid):
        if uid not in self.group:
            return -1
        else:
            return self.group[uid]


class UserVoteCluster(Cluster):

    def __init__(self, records):
        Cluster.__init__(self, records)
        vote, cnt = {}, {}
        for r in records:
            if r.user not in vote:
                vote[r.user] = 0
                cnt[r.user] = 0
            vote[r.user] += r.rate
            cnt[r.user] += 1
        # 按照物品平均评分进行分组
        for user, v in vote.items():
            c = v / (cnt[user] * 1.0)
            self.group[user] = int(c * 2)

    def get_group(self, uid):
        if uid not in self.group:
            return -1
        else:
            return self.group[uid]

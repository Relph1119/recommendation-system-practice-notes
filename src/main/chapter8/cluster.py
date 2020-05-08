#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: cluster.py
@time: 2020/5/8 10:43
@project: recommendation-system-practice-notes
@desc: Cluster基类
"""


class Cluster:

    def __init__(self, records):
        self.group = {}

    def get_group(self, i):
        return 0


class IdCluster(Cluster):

    def __init__(self, records):
        Cluster.__init__(self, records)

    def get_group(self, i):
        return i

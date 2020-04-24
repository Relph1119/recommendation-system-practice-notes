#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: tag_based_tfidf_plus.py
@time: 2020/4/23 19:13
@project: recommendation-system-practice-notes
@desc: TagBasedTFIDF++算法
"""

from math import log1p
from operator import itemgetter

from main.chapter4.tag_based_tfidf import TagBasedTFIDF


class TagBasedTFIDFPlus(TagBasedTFIDF):

    def __init__(self):
        super().__init__()
        self.item_user_count = dict()

    def _build_matrix(self, train):
        super()._build_matrix(train)
        for user_id, item_id, tag_id in train:
            self.item_user_count.setdefault(item_id, set())
            self.item_user_count[item_id].add(user_id)

        self.item_user_count = {item_id: len(users) for item_id, users in self.item_user_count.items()}

    def _recommend_user(self, user):
        user_tags = self.user_tag[user]
        recommend_dict = dict()

        user_buys = self.user_item[user]  # 用户已经买过的商品
        for tag_id, tag_count in user_tags.items():
            for item_id, item_count in self.tag_item[tag_id].items():
                # 如果已经买过则不加入推荐名单中
                if item_id in user_buys:
                    continue

                recommend_dict.setdefault(item_id, 0)
                recommend_dict[item_id] += (tag_count * item_count /
                                            (log1p(self.tag_user_count[tag_id]) * log1p(self.item_user_count[item_id])))

        recommends = sorted(recommend_dict.items(), key=itemgetter(1), reverse=True)
        return recommends

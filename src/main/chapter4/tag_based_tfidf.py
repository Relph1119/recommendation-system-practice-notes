#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: tag_based_tfidf.py
@time: 2020/4/23 16:40
@project: recommendation-system-practice-notes
@desc: TagBasedTFIDF算法
"""

from math import log1p
from operator import itemgetter

from main.chapter4.simple_tag_based import SimpleTagBased


class TagBasedTFIDF(SimpleTagBased):
    """基于TFIDF的改进版"""

    def _build_matrix(self, train):
        super()._build_matrix(train)

        self.tag_user_count = dict()
        for user_id, item_id, tag_id in train:
            self.tag_user_count.setdefault(tag_id, set())
            self.tag_user_count[tag_id].add(user_id)

        self.tag_user_count = {tag_id: len(users) for tag_id, users in self.tag_user_count.items()}

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
                recommend_dict[item_id] += tag_count * item_count / log1p(self.tag_user_count[tag_id])

        recommends = sorted(recommend_dict.items(), key=itemgetter(1), reverse=True)
        return recommends

#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: main.py
@time: 2020/4/20 19:14
@project: recommendation-system-practice-notes
@desc:
"""
import os
import sys

import pandas as pd

from main.chapter5.recent_popularity import RecentPopular
from main.util import delicious_reader, metric

PROJECT_ROOT = os.path.dirname(sys.path[0])
user_bookmark_path = os.path.join(PROJECT_ROOT, "data/delicious-2k/user_taggedbookmarks-timestamps.dat")
bookmarks_path = os.path.join(PROJECT_ROOT, "data/delicious-2k/bookmarks.dat")

original_dataset = delicious_reader.load_data(bookmarks_path, user_bookmark_path)

train_dataset, test_dataset = delicious_reader.split_data(
    delicious_reader.filter_dataset(original_dataset, "nytimes.com"))
model = RecentPopular(train_dataset)
model.fit()


def train_popularity(train_dataset):
    """计算训练集的流行度"""
    train_popularity = dict()
    for user, items_timestamp in train_dataset.items():
        for item, _ in items_timestamp:
            train_popularity.setdefault(item, 0)
            train_popularity[item] += 1
    return train_popularity


def evaluate(model, original_dataset, train_dataset, test_dataset, N, K=None):
    """
    评估模型
    :param N: 推荐的商品个数
    :param K: 搜索邻近的用户个数
    :return: 精确率(precision), 召回率(recall), 覆盖率(coverage), 流行度(popularity)
    """
    test_dataset = get_all_items(test_dataset)

    recommens = model.recommend_users(train_dataset.keys(), N=N, K=K)
    all_items = delicious_reader.all_items(original_dataset)
    item_popularity = train_popularity(train_dataset)

    recall = metric.recall(recommends=recommens, tests=test_dataset)
    precision = metric.precision(recommends=recommens, tests=test_dataset)

    return precision, recall


def get_all_items(dataset):
    user_items = dict()
    for user, items_timestamp in dataset.items():
        items = []
        for item, _ in items_timestamp:
            items.append(item)
        user_items[user] = items
    return user_items


metric_value = list()

N_list = [i for i in range(10, 110, 10)]

for N in N_list:
    single_eval = evaluate(model, original_dataset, train_dataset, test_dataset, N, K=None)
    metric_value.append(single_eval)

a = pd.DataFrame(
    data=metric_value,
    index=N_list,
    columns=["Precision", "Recall"]
)

print(a)

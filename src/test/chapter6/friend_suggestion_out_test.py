#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: friend_suggestion_out_test.py
@time: 2020/4/20 19:14
@project: recommendation-system-practice-notes
@desc:
"""
import os
import sys

import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from main.chapter6.friend_suggestion_out import FriendSuggestionOut
from main.util import slashdot_reader, metric

PROJECT_ROOT = os.path.dirname(sys.path[0])
slashdot_path = os.path.join(PROJECT_ROOT, "../data/soc-Slashdot0902/Slashdot0902.txt")

# 加载数据集
slashdot_dataset = slashdot_reader.load_data(slashdot_path)


def evaluate(model, test_dataset, N):
    """
    评估模型
    :param N: 推荐的商品个数
    :param K: 搜索邻近的用户个数
    :return: 精确率(precision), 召回率(recall)
    """
    recommens = model.recommend_users(test_dataset.keys(), N=N)

    recall = metric.recall(recommends=recommens, tests=test_dataset)
    precision = metric.precision(recommends=recommens, tests=test_dataset)

    return precision, recall


# 对数据集进行划分
M = 10
N = 10
kf = KFold(n_splits=M, shuffle=True, random_state=1)
precisions = []
recalls = []
metric_value = []
for train_index, test_index in kf.split(slashdot_dataset):
    test_dataset = slashdot_dataset.iloc[test_index]
    train_dataset = slashdot_dataset.iloc[train_index]
    train_dataset, test_dataset = slashdot_reader.convert_dict(train_dataset, test_dataset)
    # 模型训练
    model = FriendSuggestionOut(train_dataset)
    precision, recall = evaluate(model, test_dataset[0], N)
    precisions.append(precision)
    recalls.append(recall)
metric_value.append((np.average(precisions), np.average(recalls)))

result_df = pd.DataFrame(
    data=metric_value,
    index=['FriendSuggestionOut'],
    columns=['Precision', 'Recall']
)
print(result_df)

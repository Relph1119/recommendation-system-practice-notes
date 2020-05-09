#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: friend_suggestion_in_test.py
@time: 2020/4/20 19:14
@project: recommendation-system-practice-notes
@desc:
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from main.chapter3.age_most_popular import AgeMostPopular
from main.chapter3.country_most_popular import CountryMostPopular
from main.chapter3.demographic_most_popular import DemographicMostPopular
from main.chapter3.gender_most_popular import GenderMostPopular
from main.chapter3.most_popular import MostPopular
from main.util import lastfm_reader, metric

PROJECT_ROOT = os.path.dirname(sys.path[0])

data_path = os.path.join(PROJECT_ROOT, "../data/lastfm-360K/usersha1-artmbid-artname-plays.csv")
profile_path = os.path.join(PROJECT_ROOT, "../data/lastfm-360K/usersha1-profile.csv")

# 加载数据集
data, profile = lastfm_reader.load_data(data_path, profile_path)


def evaluate(model, train_dataset, test_dataset, N):
    """
    评估模型
    :param N: 推荐的商品个数
    :param K: 搜索邻近的用户个数
    :return: 精确率(precision), 召回率(recall)
    """
    recommens = model.recommend_users(test_dataset.keys(), N=N)
    all_items = lastfm_reader.get_all_items(train_dataset, test_dataset)

    recall = metric.recall(recommends=recommens, tests=test_dataset)
    precision = metric.precision(recommends=recommens, tests=test_dataset)
    coverage = metric.coverage(recommends=recommens, all_items=all_items)

    return precision, recall, coverage


def train(kf, data, popular_model):
    precisions = []
    recalls = []
    coverages = []
    for train_index, test_index in kf.split(data):
        test_dataset = data.iloc[test_index]
        train_dataset = data.iloc[train_index]
        train_dataset, test_dataset = lastfm_reader.convert_dict(train_dataset, test_dataset)
        # 模型训练
        model = popular_model(train_dataset, profile)
        model.fit()
        precision, recall, coverage = evaluate(model, train_dataset, test_dataset, N)
        precisions.append(precision)
        recalls.append(recall)
        coverages.append(coverage)
    return np.average(precisions), np.average(recalls), np.average(coverages)


# 对数据集进行划分
M = 10
N = 10
kf = KFold(n_splits=M, shuffle=True, random_state=1)
popular_group = [MostPopular, GenderMostPopular, AgeMostPopular, CountryMostPopular, DemographicMostPopular]
metric_value = []
index_names = []

for popular_model in popular_group:
    index_names.append(popular_model.__name__)
    metric_value.append(train(kf, data, popular_model))

result_df = pd.DataFrame(
    data=metric_value,
    index=index_names,
    columns=['Precision', 'Recall', 'Coverage']
)
print(result_df)

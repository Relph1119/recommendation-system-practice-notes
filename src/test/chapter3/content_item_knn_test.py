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

from main.chapter3.content_item_knn import ContentItemKNN
from main.util import metric, movielen_reader


def evaluate(model, train_dataset, test_dataset, N, K):
    """
    评估模型
    :param N: 推荐的商品个数
    :param K: 搜索邻近的用户个数
    :return: 精确率(precision), 召回率(recall), 覆盖率(coverage)
    """
    recommens = model.recommend_users(test_dataset.keys(), N=N, K=K)
    all_items = movielen_reader.get_all_items(train_dataset, test_dataset)
    item_popularity = train_popularity(train_dataset)

    recall = metric.recall(recommends=recommens, tests=test_dataset)
    precision = metric.precision(recommends=recommens, tests=test_dataset)
    coverage = metric.coverage(recommends=recommens, all_items=all_items)
    popularity = metric.popularity(item_popular=item_popularity, recommends=recommens)

    return precision, recall, coverage, popularity


def train(kf, data, contents, N, K):
    precisions = []
    recalls = []
    coverages = []
    popularitys = []
    for train_index, test_index in kf.split(data):
        test_dataset = data.iloc[test_index]
        train_dataset = data.iloc[train_index]
        train_dataset, test_dataset = movielen_reader.convert_dict(train_dataset, test_dataset)
        # 模型训练
        model = ContentItemKNN(train_dataset, contents)
        model.fit()
        precision, recall, coverage, popularity = evaluate(model, train_dataset, test_dataset, N, K)
        precisions.append(precision)
        recalls.append(recall)
        coverages.append(coverage)
        popularitys.append(popularity)
    return np.average(precisions), np.average(recalls), np.average(coverages), np.average(popularitys)


def train_popularity(train_dataset):
    """计算训练集的流行度"""
    train_popularity = dict()
    for user, items in train_dataset.items():
        for item in items:
            train_popularity[item] = train_popularity.get(item, 0) + 1
    return train_popularity


if __name__ == '__main__':
    PROJECT_ROOT = os.path.dirname(sys.path[0])

    ratings_path = os.path.join(PROJECT_ROOT, "data/ml-1m/ratings.dat")
    movies_path = os.path.join(PROJECT_ROOT, "data/ml-1m/movies.dat")

    # 加载数据集
    ratings_dataset = movielen_reader.load_data(ratings_path, ['user', 'item', 'rating', 'timestamp'])
    movies_dataset = movielen_reader.load_data(movies_path, ['item', 'title', 'genres'])
    movies_dataset['genres'] = movies_dataset['genres'].str.split('|')
    ratings_dataset = ratings_dataset[['user', 'item', 'rating']]

    contents = {row['item']: row['genres'] for _, row in movies_dataset.iterrows()}

    M = 10
    N = 10
    K = 10
    # K-Fold模型训练
    kf = KFold(n_splits=M, shuffle=True, random_state=1)
    metric_value = [train(kf, ratings_dataset, contents, N, K)]
    # 得到指标
    result_df = pd.DataFrame(
        data=metric_value,
        index=[ContentItemKNN.__name__],
        columns=['Precision', 'Recall', 'Coverage', 'Popularity']
    )

    print(result_df)

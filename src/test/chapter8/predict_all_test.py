#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: main.py
@time: 2020/4/20 19:14
@project: recommendation-system-practice-notes
@desc:
"""
import math
import os
import sys

import pandas as pd

from main.chapter8.cluster import Cluster, IdCluster
from main.chapter8.item_cluster import ItemPopularityCluster, ItemVoteCluster
from main.chapter8.predict_all import PredictAll
from main.chapter8.user_cluster import UserActivityCluster, UserVoteCluster
from main.util import movielen_reader

PROJECT_ROOT = os.path.dirname(sys.path[0])
movie_len_path = os.path.join(PROJECT_ROOT, "../data/ml-1m/ratings.dat")

# 加载数据集
movie_len_dataset = movielen_reader.load_data(movie_len_path)
train_dataset, test_dataset = movielen_reader.split_data(movie_len_dataset)


def RMSE(records):
    """计算RMSE
        @param records: 预测评价与真实评价记录的一个list
        @return: RMSE
    """
    numerator = sum([(r.rate - r.predict) ** 2 for r in records])
    denominator = float(len(records))
    return math.sqrt(numerator / denominator)


UserGroups = [Cluster, IdCluster, Cluster, UserActivityCluster, UserActivityCluster, Cluster, IdCluster,
              UserActivityCluster, UserVoteCluster, UserVoteCluster, Cluster, IdCluster, UserVoteCluster]
ItemGroups = [Cluster, Cluster, IdCluster, Cluster, IdCluster, ItemPopularityCluster, ItemPopularityCluster,
              ItemPopularityCluster, Cluster, IdCluster, ItemVoteCluster, ItemVoteCluster, ItemVoteCluster]

metric = []
for i in range(len(UserGroups)):
    user_group = UserGroups[i]
    item_group = ItemGroups[i]
    model = PredictAll(train_dataset, test_dataset, user_group, item_group)
    train_dataset = model.fit()
    test_dataset = model.predict(test_dataset)

    train_metric = RMSE(train_dataset)
    test_metric = RMSE(test_dataset)
    metric.append((user_group.__name__, item_group.__name__, train_metric, test_metric))

result_df = pd.DataFrame(
    data=metric,
    index=[i for i in range(len(UserGroups))],
    columns=['UserGroup', 'ItemGroup', 'TrainRMSE', 'TestRMSE']
)
print(result_df)

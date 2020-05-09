#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: lastfm_reader.py
@time: 2020/5/8 22:23
@project: recommendation-system-practice-notes
@desc: lastfm 360K数据集的读取
"""

import numpy as np
import pandas as pd


def load_data(data_path, profile_path, sample=5000):
    """
    加载数据集
    :param data_path: usersha1-artmbid-artname-plays数据集路径
    :param profile_path: usersha1-profile.csv数据集路径
    :param sample: 样本数
    :return:
    """
    data_dataset = pd.read_csv(data_path, header=None, names=['user', 'item', 'artist_name', 'plays'])
    profile_dataset = pd.read_csv(profile_path, header=None, names=['user', 'gender', 'age', 'country', 'signup'])
    users = set(profile_dataset['user'][:sample])
    data = data_dataset[data_dataset['user'].isin(users)]
    profile = profile_dataset[profile_dataset['user'].isin(users)]
    profile['age'] = profile['age'].fillna(-1).astype(int)

    profile = {k: profile[profile.user == k][['gender', 'age', 'country']].reset_index(drop=True).to_dict(orient='records')[0] for k in users}
    return data, profile


def convert_dict(train_dataset, test_dataset):
    train_dataset = _convert(train_dataset)
    test_dataset = _convert(test_dataset)
    return train_dataset, test_dataset


def _convert(data):
    """
    处理成字典的形式，user->set(items)
    :param data: 需要处理的数据集
    :return:
    """
    # 当前用户指向的用户
    data_dict = {}
    for u, v in zip(data['user'], data['item']):
        if u not in data_dict:
            data_dict[u] = set()
        if v is not np.nan:
            data_dict[u].add(v)
    data_dict = {k: list(data_dict[k]) for k in data_dict}
    return data_dict


def get_all_items(train_dataset, test_dataset):
    """
    返回所有的item
    :param test_dataset: 训练数据集
    :param train_dataset: 测试数据集
    :return:
    """
    all_items = set()

    def add_items(dataset, all_items):
        for user, items in dataset.items():
            items = set(items)
            all_items = all_items.union(items)
        return all_items

    all_items = add_items(train_dataset, all_items)
    all_items = add_items(test_dataset, all_items)

    return all_items


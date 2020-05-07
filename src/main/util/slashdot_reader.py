#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: slashdot_reader.py
@time: 2020/5/7 16:34
@project: recommendation-system-practice-notes
@desc: Slashdot0902数据集的读取
"""

import pandas as pd


def load_data(slashdot_path, sample=100000):
    """
    加载数据集
    :param slashdot_path: slashdot数据集路径
    :param sample: 样本数
    :return:
    """
    slashdot_dataset = pd.read_table(slashdot_path, skiprows=[0, 1, 2, 3], sep='\t',
                                     header=None, engine='python',
                                     names=['FromNodeId', 'ToNodeId'])
    if sample == -1:
        return slashdot_dataset
    else:
        return slashdot_dataset[:sample]


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
    # 指向当前用户的用户
    data_dict_t = {}
    for u, v in zip(data['FromNodeId'], data['ToNodeId']):
        if u not in data_dict:
            data_dict[u] = set()
        data_dict[u].add(v)
        if v not in data_dict_t:
            data_dict_t[v] = set()
        data_dict_t[v].add(u)
    data_dict = {k: list(data_dict[k]) for k in data_dict}
    data_dict_t = {k: list(data_dict_t[k]) for k in data_dict_t}
    return data_dict, data_dict_t

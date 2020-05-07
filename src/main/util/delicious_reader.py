#!/usr/bin/python3
# coding=utf-8
"""
Created on 2018年6月2日
@author: qcymkxyc
@desc: Delicious数据集的读取
"""

import pandas as pd


def filter_dataset(dataset, site):
    dataset = dataset[dataset['urlPrincipal'].str.find(site) != -1] \
        .drop(labels=['urlPrincipal', 'tagID'], axis=1).drop_duplicates()

    data = {}
    for index, row in dataset.iterrows():
        user_id = row['userID']
        bookmark_id = row['bookmarkID']
        timestamp = row['timestamp'] // 1000
        if user_id not in data:
            data[user_id] = set()
        data[user_id].add((bookmark_id, timestamp))

    data = {k: list(sorted(list(data[k]), key=lambda x: x[1], reverse=True)) for k in data}
    return data


def load_data(bookmark_path, user_bookmark_path):
    user_bookmark_dataset = pd.read_table(user_bookmark_path, sep='\t', engine='python')
    bookmark_dataset = pd.read_table(bookmark_path, sep='\t', engine='python')
    bookmark_dataset.rename(columns={'id': 'bookmarkID'}, inplace=True)
    dataset = pd.merge(user_bookmark_dataset, bookmark_dataset, how='left', on=['bookmarkID'])
    dataset = dataset[['userID', 'bookmarkID', 'tagID', 'urlPrincipal', 'timestamp']]
    return dataset


def split_data(dataset, bookmark_path=None, user_bookmark_path=None, site="", load_data_flag=False):
    """
    对每个用户按照时间进行从前到后的排序，取最后一个时间的item作为测试集
    :param dataset: 数据集
    :param load_data_flag: 是否加载数据
    :param bookmark_path: bookmark数据集的文件路径
    :param user_bookmark_path: user_taggedbookmarks-timestamps数据集的文件路径
    :param site: 网站
    :return: train_dataset, test_dataset训练集和测试集
        {userID:[(bookmarkID, timestamp),...],...}
    """
    if load_data_flag:
        if bookmark_path is None:
            print("请输入user_taggedbookmarks-timestamps数据集的文件路径")
        if user_bookmark_path is None:
            print("请输入bookmark数据集的文件路径")
        dataset = filter_dataset(load_data(bookmark_path, user_bookmark_path), site)

    train_dataset, test_dataset = dict(), dict()
    for user in dataset:
        if user not in train_dataset:
            train_dataset[user] = list()
            test_dataset[user] = list()
        data = dataset[user]
        train_dataset[user].extend(data[1:])
        test_dataset[user].append(data[0])

    return train_dataset, test_dataset


def get_all_items(dataset):
    """
    返回所有的bookmark
    :param dataset: 数据集
    :return:
    """
    user_items = dict()
    for user, items_timestamp in dataset.items():
        items = []
        for item, _ in items_timestamp:
            items.append(item)
        user_items[user] = items
    return user_items

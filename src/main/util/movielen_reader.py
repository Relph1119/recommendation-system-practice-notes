#! /usr/bin/python3
# coding=utf-8
"""
Created on 2018年6月10日

@author: qcymkxyc
@desc: MovieLen数据集的读取
"""
import random


def load_file(filename):
    """
    根据文件名载入数据
    """
    with open(filename, "r") as f:
        for line in f:
            yield line


def read_rating_data(path="../../data/ml-1m/ratings.dat", train_rate=1., seed=1):
    """
    载入评分数据
    :param path: 文件路径
    :param train_rate: 训练集所占整个数据集的比例，默认为1，表示所有的返回数据都是训练集
    :param seed:
    :return: (训练集，测试集)
    """
    train_dataset = list()
    test_dataset = list()

    random.seed(seed)
    for line in load_file(filename=path):
        user, movie, rating, _ = line.split('::')
        if random.random() < train_rate:
            train_dataset.append([int(user), int(movie), int(rating)])
        else:
            test_dataset.append([int(user), int(movie), int(rating)])
    return train_dataset, test_dataset


def all_items(path="../../data/ml-1m/ratings.dat"):
    """返回所有的movie"""
    items = set()
    for line in load_file(filename=path):
        _, movie, _, _ = line.split("::")
        items.add(movie)
    return items

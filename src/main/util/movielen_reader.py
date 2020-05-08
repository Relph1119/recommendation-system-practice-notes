#! /usr/bin/python3
# coding=utf-8
"""
Created on 2018年6月10日

@author: qcymkxyc
@desc: MovieLen数据集的读取
"""
import random

import pandas as pd
from sklearn.utils import shuffle as reset


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


class MovieLenData:

    def __init__(self, user, item, rate, predict=0.0):
        self.user = user
        self.item = item
        self.rate = rate
        self.predict = predict


def load_data(file_path):
    dataset = pd.read_table(file_path, sep='::',
                            header=None, engine='python',
                            names=['user', 'item', 'rating', 'timestamp'])
    return dataset


def split_data(dataset, test_size=0.1, shuffle=False, random_state=None):
    train_dataset, test_dataset = _train_test_split(dataset, test_size, shuffle, random_state)
    train_dataset = train_dataset[['user', 'item', 'rating']]
    test_dataset = test_dataset[['user', 'item', 'rating']]
    train_dataset = [MovieLenData(*d) for d in train_dataset.values]
    test_dataset = [MovieLenData(*d) for d in test_dataset.values]

    return train_dataset, test_dataset


def _train_test_split(data, test_size=0.3, shuffle=True, random_state=None):
    """Split DataFrame into random train and test subsets

    Parameters
    ----------
    data : pandas dataframe, need to split dataset.

    test_size : float
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    shuffle : boolean, optional (default=None)
        Whether or not to shuffle the data before splitting. If shuffle=False
        then stratify must be None.
    """

    if shuffle:
        data = reset(data, random_state=random_state)

    train_size = 1 - test_size
    test_dataset = data[int(len(data) * train_size):].reset_index(drop=True)
    train_dataset = data[:int(len(data) * train_size)].reset_index(drop=True)

    return train_dataset, test_dataset

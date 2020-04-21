#! /usr/bin/python3
# coding=utf-8

import math


def RMSE(records):
    """计算RMSE
        @param records: 预测评价与真实评价记录的一个list
        @return: RMSE
    """
    numerator = sum([(pred_rating - actual_rating) ** 2 for pred_rating, actual_rating in records])
    denominator = float(len(records))
    return math.sqrt(numerator / denominator)


def MAE(records):
    """计算MAE
        @param records: 预测评价与真实评价记录的一个list
        @return: RMSE
    """
    numerator = sum([abs(pred_rating - actual_rating) for pred_rating, actual_rating in records])
    denominator = float(len(records))
    return numerator / denominator


def MSE(records):
    """计算MSE
        @param records: 预测评价与真实评价记录的一个list
        @return: MSE
    """
    numerator = sum([(pred_rating - actual_rating) ** 2 for pred_rating, actual_rating in records])
    denominator = float(len(records))
    return numerator / denominator


def precision(recommends, tests):
    """
    计算Precision
    :param recommends: 给用户推荐的商品，recommends为一个dict，格式为 { user_id : 推荐的物品 }
    :param tests: 测试集，同样为一个dict，格式为 { userID : 实际发生事务的物品 }
    :return: Precision
    """
    n_union = 0.
    user_sum = 0.
    for user_id, items in recommends.items():
        recommend_set = set(items)
        test_set = set(tests[user_id])
        n_union += len(recommend_set & test_set)
        user_sum += len(test_set)

    return n_union / user_sum


def recall(recommends, tests):
    """
    计算Recall召回率
    :param recommends: 给用户推荐的商品，recommends为一个dict，格式为{ user_id : 推荐的物品 }
    :param tests: 测试集，同样为一个dict，格式为{ user_id : 实际发生行为的物品 }
    :return:
    """
    n_union = 0.
    recommend_sum = 0.
    for user_id, items in recommends.items():
        recommend_set = set(items)
        test_set = set(tests[user_id])
        n_union += len(recommend_set & test_set)
        recommend_sum += len(recommend_set)

    return n_union / recommend_sum


def coverage(recommends, all_items):
    """
    计算覆盖率
    :param recommends:
    :param all_items: 所有的物品，为list或set类型
    :return:
    """
    recommend_items = set()
    for _, items in recommends.items():
        for item in items:
            recommend_items.add(item)
    return len(recommend_items) / len(all_items)


def popularity(item_popular, recommends):
    """
    计算流行度
    :param item_popular: 商品流行度　dict形式{ item_id : popularity}
    :param recommends: 给用户推荐的商品，recommends为一个dict，格式为{ user_id : 推荐的物品 }
    :return: 平均流行度
    """
    popularity_value = 0.  # 流行度
    n = 0.
    for _, items in recommends.items():
        for item in items:
            popularity_value += math.log(1. + item_popular.get(item, 0.))
            n += 1
    return popularity_value / n

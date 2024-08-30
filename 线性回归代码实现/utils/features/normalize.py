"""Normalize features"""

import numpy as np


def normalize(features):
    #标准化即(x-u)/std((样本值-均值)/标准差)类似于化为标准正太分布，x-u使得样本都集中在原点附近，/std使得其特征值幅度没那么大
    features_normalized = np.copy(features).astype(float)

    # 计算均值
    features_mean = np.mean(features, 0)

    # 计算标准差
    features_deviation = np.std(features, 0)

    # 标准化操作
    if features.shape[0] > 1:
        features_normalized -= features_mean

    # 防止除以0
    features_deviation[features_deviation == 0] = 1
    features_normalized /= features_deviation

    return features_normalized, features_mean, features_deviation

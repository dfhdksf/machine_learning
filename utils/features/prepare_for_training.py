"""Prepares the dataset for training"""

import numpy as np
from .normalize import normalize
from .generate_sinusoids import generate_sinusoids
from .generate_polynomials import generate_polynomials


def prepare_for_training(data, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):

    # 计算样本总数
    num_examples = data.shape[0]

    data_processed = np.copy(data)

    # 预处理
    features_mean = 0
    features_deviation = 0
    data_normalized = data_processed
    if normalize_data:#true，进行标准化
        (
            data_normalized,
            features_mean,
            features_deviation
        ) = normalize(data_processed)

        data_processed = data_normalized

    # 特征变换sinusoidal
    if sinusoid_degree > 0:
        sinusoids = generate_sinusoids(data_normalized, sinusoid_degree)
        data_processed = np.concatenate((data_processed, sinusoids), axis=1)

    # 特征变换polynomial
    if polynomial_degree > 0:
        polynomials = generate_polynomials(data_normalized, polynomial_degree, normalize_data)
        data_processed = np.concatenate((data_processed, polynomials), axis=1)

    # 加一列1：y=θ1x1+...+θn*1，由于有num_examples个样本，所以用一列1
    #np.hstack()把多个数组沿水平方向（即按列的方向）合并在一起,最终，data_processed 将被更新为一个新的数组，
    # 其第一列是一列全为 1 的数值，后面跟着原始 data_processed 中的所有列。
    data_processed = np.hstack((np.ones((num_examples, 1)), data_processed))

    return data_processed, features_mean, features_deviation

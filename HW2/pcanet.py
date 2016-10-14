#!/usr/bin/python
"""
README: CSE841 HW2

This is an implementation of a PCA Net.
Written by Adi Mathew.
10/13/16

This program uses Python.
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d


def plotvec(mx, colors, symbol, labels, title="Title"):
    """
    Plots a given 3D matrix.
    Each element of mx is expected to be 3 x n.
    Number of elements in mx, colors and labels are expected to be same.
    """
    plt.rcParams["toolbar"] = "None"
    fig = plt.figure(figsize=(9, 9), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    plt.rcParams["legend.fontsize"] = 12
    for i, v in enumerate(mx):
        ax.plot(
                v[0, :], v[1, :], v[2, :],
                symbol,
                markersize=8,
                color=colors[i],
                alpha=1.0,
                label=labels[i])

    plt.title(title)
    ax.legend(loc='upper right')

    plt.show()

mu_vec1 = np.array([0, 0, 0])
cov_mat1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20).T
assert class1_sample.shape == (3, 20), "The matrix has not the dimensions 3x20"

mu_vec2 = np.array([1, 1, 1])
cov_mat2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20).T
assert class1_sample.shape == (3, 20), "The matrix has not the dimensions 3x20"

plotvec((class1_sample, class2_sample),
        ['blue', 'red'],
        'o',
        labels=["class1", "class2"],
        title="Samples for class 1 and class 2")

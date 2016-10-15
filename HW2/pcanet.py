#!/usr/local/bin/python3
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

# TODO: Create argument parser for final deliverable.


def plotvec(mx, colors, symbol, labels, title="Title"):
    """
    Plots a given list of 3D vectors
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
                alpha=0.75,
                label=labels[i])

    plt.title(title)
    ax.legend(loc='upper right')

    plt.show()

"""
# ------------------------------------------------------------------------------
# This is debug code
np.random.seed(234134784384739784)
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
# ------------------------------------------------------------------------------
"""


def scalenorm(vec, delta=128):
    """
    Performs scale normalization with a given delta value.
    Returns the normalized vector as a list of floats.
    """
    for x in np.nditer(vec, op_flags=["readwrite"]):
        if x < delta:
            x[...] = 0

    isz = not np.any(vec)
    normvec = []

    if not isz:
        nmin = np.amin(vec)
        nmax = np.amax(vec)
        for x in np.nditer(vec):
            y = (x - nmin)/(nmax - nmin)
            normvec.append(y)

    return normvec


def amnesicmean(mcurrent, tcurrent, xinput, t1, t2, r, c):
    """
    Calculates a weighted average of inputs so far.
    Parameters t1, t2, r and c are used for u(t) calculation.
    """
    mcurrent = np.array(mcurrent)
    xinput = np.array(xinput)
    ut = None
    if tcurrent < t1:
        ut = 0
    elif tcurrent < t2 and tcurrent >= t1:
        ut = c * (tcurrent - t1)/(t2 - t1)
    elif tcurrent >= t2:
        ut = c + (tcurrent - t2)/r

    a = (tcurrent - 1 - ut)/tcurrent
    b = (1 + ut)/tcurrent

    return (a * mcurrent) + (b * xinput)


def meannormal(xinput, mean):
    """
    Returns the scatter vector, u for x.
    Use this along with amnesicmean().
    """
    return np.subtract(xinput, mean)


def vec2im(vec, xdim=88, ydim=64):
    """
    Takes a numpy array(vec) and it's unflattened dimensions
    and displays the image contained.
    """
    im = vec.reshape(xdim, ydim)
    plt.rcParams["toolbar"] = "None"
    fig = plt.figure(figsize=(6, 6), facecolor="white")
    plt.imshow(im, plt.cm.gray)
    plt.colorbar()
    plt.show()

# ------------------------------------------------------------------------------


def getvecnames(file, folder="803Fall07/"):
    fnames = []
    with open(folder+file, "r") as data:
        # To store the total number of people and testcases
        n, tot = 0, 0

        for i, line in enumerate(data):
            # Check if it is an empty line
            if line.strip():
                # Store the first two total values
                if i == 0:
                    n = int(line)
                elif i == 1:
                    tot = int(line)
                # Only after n and tot are stored, append filenames using them.
                if i > (n - 1) + 2 and n > 0 and tot > 0:
                    line = line.strip()
                    fnames.append(folder+line)
    return fnames

# ------------------------------------------------------------------------------

dbg_filename = "803Fall07/benA3.raw.face"
"""
data = None
with open(dbg_filename, "rb") as bin:
    data = bytearray(bin.read())
    ndata = np.frombuffer(data, dtype='u1')
    # print(np.array_equal(data, ndata))

    dbg_data_1 = np.array([32, 64, 128], dtype='u1')
    dbg_data_2 = np.array([64, 128, 255], dtype='u1')
    # dbg_data = np.empty(3, dtype='u1')
    # dbg_data = np.concatenate((dbg_data, dbg_data_1), axis=0)
    # dbg_data = np.concatenate((dbg_data, dbg_data_2), axis=0)

    # print(dbg_data)

    norm = scalenorm(ndata, 0)

    dbg_data_1 = scalenorm(dbg_data_1, 0)
    dbg_data_2 = scalenorm(dbg_data_2, 0)

    # Update (i + 1) the tcurrent param below when placed in for loop
    mean = amnesicmean(dbg_data_1, 2, dbg_data_2, 5, 25, 100, 2)
    print(mean)
    print(dbg_data_1, meannormal(dbg_data_1, mean))

    for x in dbg_data_1:
        print(x, end=' ')
    print(len(data))

    dbg_data_1 = 255*np.array(norm, dtype='f')
    vec2im(dbg_data_1)
"""
files = getvecnames("traininglist.txt", "803Fall07/")
meanvec = None
for i, rawvec in enumerate(files):
    with open(rawvec, "rb") as bin:
        data = bytearray(bin.read())
        ndata = np.frombuffer(data, dtype='u1')
        norm = scalenorm(ndata)
        if i == 0:
            meanvec = norm
            continue
        else:
            meanvec = amnesicmean(meanvec, i+1, norm, 5, 25, 100, 2)
disp_meanvec = 255 * np.array(meanvec, dtype='f')
vec2im(disp_meanvec)

"""
# Use this for writing binary output to file.
with open("debug.out", "wb") as bin:
    bin.write(data)
"""

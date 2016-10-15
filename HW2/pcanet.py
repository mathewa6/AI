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

    u = u_t(tcurrent, t1, t2, r, c)

    a = (tcurrent - 1 - u)/tcurrent
    b = (1 + u)/tcurrent

    return (a * mcurrent) + (b * xinput)


def u_t(t, t1, t2, r, c):
    """
    This returns the value of the amnesic parameter for a given t.
    """
    ut = None
    if t < t1:
        ut = 0
    elif t < t2 and t >= t1:
        ut = c * (t - t1)/(t2 - t1)
    elif t >= t2:
        ut = c + (t - t2)/r

    return ut


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
    # vec = np.array(vec)
    im = vec.reshape(xdim, ydim)
    plt.rcParams["toolbar"] = "None"
    fig = plt.figure(figsize=(6, 6), facecolor="white")
    plt.imshow(im, plt.cm.gray)
    plt.colorbar()
    plt.show()

# ------------------------------------------------------------------------------


def getvecnames(file, folder="803Fall07/"):
    """
    Returns a list of binary image vectors from a given text file.
    folder parameter is appended to each filename in the return list.
    """
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
allinput = []
scat = []
pcv = []

for i, rawvec in enumerate(files):
    if i == 4:
         break

    with open(rawvec, "rb") as bin:
        data = bytearray(bin.read())
        ndata = np.frombuffer(data, dtype='u1')
        allinput.append(ndata)
        norm = scalenorm(ndata)

        if i == 0:
            meanvec = norm
            continue
        else:
            t1 = 25
            t2 = 75
            r = 10
            c = 2
            t = i+1

            meanvec = amnesicmean(meanvec, t, norm, t1, t2, r, c)
            scat.append(meannormal(norm, meanvec))

            for j in range(1, t):
                # print(i,len(scat))
                u = scat[j-1]
                if j == i:
                    # print(np.array_equal())
                    # print(j, i)
                    pcv.append(i)
                else:
                    v = pcv[j-1]
                    norm = v / np.linalg.norm(v)
                    # a)
                    y = np.inner(u, norm)
                    # b)
                    # TODO Check if u_t should be i+1 or i
                    u_amn = u_t(t, t1, t2, r, c)
                    w1 = (t - 1 - u_amn)/t
                    w2 = (1 + u_amn)/t
                    pcv[j-1] = (w1 * v) + (w2 * y * u)
                    # c)
                    scat[j] = u - (np.inner(y, norm))

disp_meanvec = 255 * np.array(meanvec, dtype='f')
vec2im(disp_meanvec)

disp_pcv = 255 * np.array(pcv[0], dtype='f')
print(np.dot(scat[0], scat[1]))
vec2im(allinput[0])

vec2im(disp_pcv)
"""
# Use this for writing binary output to file.
with open("debug.out", "wb") as bin:
    bin.write(data)
"""

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


def scalenorm(v, delta=128):
    """
    Performs scale normalization with a given delta value.
    Returns the normalized vector as a list of floats.
    """
    vec = v
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

files = getvecnames("input.txt", "Test/")
meanvec = None
allinput = []
scat = []
pcv = []
eig = []

for t, rawvec in enumerate(files):
    # if i == 4:
    #    break

    with open(rawvec, "rb") as bin:
        data = bytearray(bin.read())
        ndata = np.frombuffer(data, dtype='u1')
        allinput.append(ndata)
        norm = scalenorm(ndata, 1)

        if t == 0:
            meanvec = norm
            continue
        else:
            t1 = 25
            t2 = 75
            r = 10
            c = 2
            x = t+1

            # meanvec = amnesicmean(meanvec, t, norm, t1, t2, r, c)
            meanvec = np.inner((t/(t+1)), meanvec) + np.inner((1/t+1), norm)
            scat.append(meannormal(norm, meanvec))

            for i in range(1, x):
                # print(i,len(scat))
                u = scat[i-1]
                if i == t:
                    # print(np.array_equal())
                    # print(j, i)
                    pcv.append(u)
                    eig.append(np.linalg.norm(u))
                else:
                    v = pcv[i-1]
                    norval = v / np.linalg.norm(v)
                    # a)
                    y = np.inner(u, norval)
                    # b)
                    # TODO Check if u_t should be i+1 or i
                    u_amn = 2  # u_t(t, t1, t2, r, c)
                    w1 = (t - 1 - u_amn)/t
                    w2 = (1 + u_amn)/t
                    pcv[i-1] = (w1 * v) + (w2 * y * u)
                    eig[i-1] = np.linalg.norm(pcv[i-1])
                    # c) j or j-1 ?
                    scat[i-1] = u - (np.inner(y, norval))

eig_pairs = [(np.abs(eig[i]), pcv[i], i) for i in range(len(eig))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)

denom = 0
for p in scat:
    denom += (1/(len(scat) - 1)) * (np.linalg.norm(p) ** 2)
print(denom, len(scat))

num = 0
for l in eig_pairs:
    num += l[0]
    ratio = l[0]/denom
    # if ratio > 0.95:
    print(l[0], l[2], ratio)

disp_meanvec = 255 * np.array(meanvec, dtype='f')
vec2im(disp_meanvec)

# print(eig_pairs)
for im in allinput:
    vec2im(im)
for im in pcv:
    disp_pcv = 255 * np.array(pcv[0], dtype='f')
    vec2im(disp_pcv)

# ------------------------------------------------------------------------------
"""
# Use this for writing binary output to file.
with open("debug.out", "wb") as bin:
    bin.write(data)
"""
"""
# ------------------------------------------------------------------------------

files = getvecnames("traininglist.txt", "803Fall07/")
meanvec = None
allinput = []
scat = None

for t, rawvec in enumerate(files):
    with open(rawvec, "rb") as bin:
        data = bytearray(bin.read())
        ndata = np.frombuffer(data, dtype='u1')
        # norm = scalenorm(ndata)
        allinput.append(ndata)

dimensions = len(allinput[0])

# meanvec = np.array([np.mean(allinput[x, :]) for x in range(dimensions)])
meanvec = [0 for x in range(dimensions)]
for v in allinput:
    for d in range(dimensions):
        meanvec[d] += v[d]
meanvec = np.array(meanvec, dtype='f')
meanvec /= dimensions

vec2im(meanvec)

scat = np.zeros((dimensions, dimensions))
for i, v in enumerate(allinput):
    scat += (v - meanvec).dot((v - meanvec).T)

eig, pcv = np.linalg.eig(scat)
print(len(eig))
vec2im(pcv[0])
# ------------------------------------------------------------------------------
"""

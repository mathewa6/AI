#!/usr/local/bin/python3
"""
README: CSE841 HW2

This is an implementation of a PCA Net.
Written by Adi Mathew.
10/13/16

This program uses Python.
"""

import os
from array import array
from time import strftime
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

from argparse import ArgumentParser as args


def getArguments():
    arguments = args(
                description="""Applies CCI PCA to an image set.
                            See CSE841 HW2 for spec..""")
    arguments.add_argument(
                            "-l", "--learn",
                            action="store",
                            type=int,
                            metavar="epochs",
                            const=1,
                            nargs='?',
                            help="Indicate whether to learn or test.")
    arguments.add_argument(
                            "-f",
                            action="store",
                            metavar="filenamelist",
                            type=str,
                            nargs=1,
                            required=True,
                            help="File containing list of filenames.")
    arguments.add_argument(
                            "-d",
                            action="store",
                            metavar="database",
                            type=str,
                            nargs=1,
                            required=True,
                            help="The PCANet binary database file.")
    arguments.add_argument(
                            "-o",
                            action="store",
                            metavar="output",
                            type=str,
                            nargs=1,
                            required=True,
                            help="Output reports file")

    ip = args.parse_args(arguments)

    return ip.learn, ip.f[0], ip.d[0], ip.o[0]


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

    return np.array(normvec, dtype='f')


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


def vec2im(
            vec,
            showim=False,
            saveim=False,
            name="Unnamed",
            folder=r"Output_" + strftime("%Y%m%d_%H%M%S") + "/",
            xdim=88,
            ydim=64):
    """
    Takes a numpy array(vec) and it's unflattened dimensions
    and displays the image contained.
    """
    # vec = np.array(vec)
    im = vec.reshape(xdim, ydim)
    plt.rcParams["toolbar"] = "None"
    plt.figure(figsize=(6, 6), facecolor="white")
    plt.imshow(im, plt.cm.gray)
    # plt.colorbar()
    if saveim:
        f = folder
        if not os.path.exists(f):
            os.makedirs(f)
        plt.savefig(f+name)

    if showim:
        plt.show()


def chunks(l, n):
    """
    Splits an array into 'n' sized bits.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]

# ------------------------------------------------------------------------------


def getvecnames(filen, folder=None):
    """
    Returns a list of binary image vectors from a given text file.
    folder parameter is appended to each filename in the return list.
    """
    fnames = []
    vnames = []

    # Clean up folder name for use with each image in file.
    if "/" in filen:
        struc = filen.split("/")
        filen = struc[-1]
        struc.remove(filen)
        folder = '/'.join(struc)
        folder += '/'

    with open(folder+filen if folder else filen, "r") as data:
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
                    vnames.append(line)
                    fnames.append(folder+line if folder else line)
    return fnames, vnames


def writefile(filen, data):
    """
    Use this for writing binary output to file.
    """
    with open(filen, "wb") as bin:
        bin.write(data)

# ------------------------------------------------------------------------------
arg = getArguments()
print(arg)
epochs = arg[0]
filename = arg[1]
db = arg[2]
op = arg[3]

files, vecs = getvecnames(filename)
allinput = []

for t, rawvec in enumerate(files):
    with open(rawvec, "rb") as bin:
        data = bytearray(bin.read())
        ndata = np.frombuffer(data, dtype='u1')
        allinput.append(ndata)

meanvec = scalenorm(allinput[0], 1)
dimensions = len(allinput[0])

if epochs:
    # Learning Phase if there is an epochs or -l flag.
    epochs = epochs[0] if type(arg[0]) is list else epochs

    scat = [np.zeros(len(allinput[0])) for e in allinput]
    yi = [np.zeros(len(allinput[0])) for e in range(len(allinput[0]))]
    pcv = [np.zeros(len(allinput[0])) for e in allinput]
    eig = [0 for e in allinput]
    k = len(allinput)
    n = len(allinput)

    for e in range(epochs):
        for t in range(len(allinput)):

            x = allinput[t]
            norm = scalenorm(x, 1)
            calct = t + 1
            meanvec = (
                        np.inner((calct/(calct+1)), meanvec) +
                        np.inner((1/calct+1), norm))
            scat[t] = (meannormal(norm, meanvec))

            for i in range(min(k, calct)):
                # print(i)
                if i == t:
                    # print("NN")
                    pcv[i] = scat[t]  # scat[t]
                else:
                    # a)
                    v = pcv[i]
                    norval = v / np.linalg.norm(v)
                    yi[i] = np.dot(scat[i], norval)
                    # print(yi[i])
                    # b)
                    u_amn = 2  # u_t(t, t1, t2, r, c)
                    w1 = (calct - 1 - u_amn)/calct
                    w2 = (1 + u_amn)/calct
                    # print(w1, w2)
                    pcv[i] = (w1 * pcv[i]) + (w2 * yi[i] * scat[i])
                    eig[i] = np.linalg.norm(pcv[i])
                    # print(eig[i])
                    # c)
                    v = pcv[i]
                    norval = v / np.linalg.norm(v)
                    # scat[i+1] = scat[i] - (np.dot(yi[i], norval))
                    # vec2im(255 * pcv[i])
    eig_pairs = [(np.abs(eig[i]), pcv[i], i) for i in range(len(eig))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    denom = 0
    for p in scat:
        denom += (1/(len(scat) - 1)) * (np.linalg.norm(p) ** 2)

    num = 0
    ratios = []
    for l in eig_pairs:
        num += l[0]
        ratios.append(num/denom)
    h = max(ratios)
    l = min(ratios)
    idx = 0
    perc = 0
    for i, r in enumerate(ratios):
        perc = (h - r)/(h - l)
        if perc < 0.95:
            idx = i
            break
    area1v = [eig_pairs[i] for i in range(idx)]

    try:
        os.remove(db)
    except OSError:
        pass

    for v in area1v:
        with open(db, 'ab') as dbv:
            print("writing")
            writeprep = array('f', v[1])
            writeprep.tofile(dbv)

    mf = "MEAN_" + ".png"
    vec2im(meanvec, False, True, mf)

    with open(op, 'w') as report:
        report.write("{:<32}: {:>32}\n".format("k", idx))
        report.write(
                    "{:<32}: {:>32.2f}%\n".format(
                                "Percentage variance",
                                perc*100.0))
        report.write("{:<32}: {:>32}\n".format("Mean filename", mf))
        for i, vp in enumerate(area1v):
            im = vp[1]
            f = vecs[vp[2]]
            nm = str(i+1) + "_MEF_" + f + ".png"
            vec2im(im, False, True, nm)
            strn = "MEF {} filename:".format(i+1)
            report.write("{:<32}: {:>48}\n".format(strn, nm))
else:
    # Testing phase if there is no epochs or -l flag.
    with open(db, 'rb') as dbr:
        y = bytearray(dbr.read())
        y = np.frombuffer(y, dtype='f')
        y = list(chunks(y, dimensions))
        for v in y:
            print(v)

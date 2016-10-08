#!/usr/bin/python
"""
This is a basic implementation of Dijkstra's and A* algorithms.
Written by Adi Mathew.
10/1/16

This program uses Python and has been tested on arctic.cse.msu.edu,
with Python 2.7.9.

All code is included in the findRoute.py file.
The file is also marked executable.
The only requirement is that the "cities" and "flightCharges" files
MUST be in the same directory as findRoute.py.

Parameters to the algorithm are passed in as argments to the command line
executable just like regular Linux scripts.
The *position* of arguments matter and are ordered
as per the HW spec: start_index, end_index, hourly_cost and future_cost.

For example, type:
./findRoute.py 0 24 10 1
to run findRoute.y with start_index = 0, end_index = 24, hourly_cost = 10 and
future_cost = 1.
"""
from __future__ import division

import sys
import math
import heapq
import datetime

from argparse import ArgumentParser as args


def getArguments():
    arguments = args(
                description="""Finds the cheapest route between 2 locations.
                            See CSE841 HW1 for spec..""")
    arguments.add_argument(
                            "start",
                            action="store",
                            metavar="start_city",
                            type=int,
                            help="Starting city for the A* algorithm")
    arguments.add_argument(
                            "end",
                            action="store",
                            metavar="end_city",
                            type=int,
                            help="Ending city for the A* algorithm")
    arguments.add_argument(
                            "hourly",
                            action="store",
                            metavar="cost_of_hour",
                            type=float,
                            help="Hourly cost for the A* algorithm")
    arguments.add_argument(
                            "future",
                            action="store",
                            metavar="future_cost",
                            type=int,
                            help="Whether or not to use the A* algorithm")
    ip = args.parse_args(arguments)

    return (ip.start, ip.end, ip.hourly, ip.future)


def getFileData(filename):
    lines = []
    with open(filename) as f:
        for line in f:
            # print(line.strip().split())
            lines.append(line.strip())
    return lines


class Node(object):
    """
    Node denotes vertices in our graph.
    City() inherits from Node.
    """
    def __init__(self):
        self.name = "Unknown"
        self.parent = None
        self.nbrs = None

    def __repr__(self):
        return "{} >>> {}".format(self.name, self.parent)


class City(Node):
    """
    City is used to traverse through the graph though it's nbr and neighbours().
    Make sure that neighbours() is called before using the self.nbr property.
    """
    def __init__(self, line, idx):
        super(City, self).__init__()
        (name, loc) = line.split()
        (x, y) = loc.strip("()").split(",")
        self.name = name
        self.x = int(x)
        self.y = int(y)
        self.hub = True if "*" in self.name else False
        self.idx = idx
        self.known = False

    def distance(self, other):
        dx = other.x - self.x
        dy = other.y - self.y
        return 4*math.sqrt(math.pow(dx, 2)+math.pow(dy, 2))

    def neighbours(self, store, pm):
        if self.nbrs:
            return self.nbrs

        n = []
        for i, p in enumerate(pm.pricemap[self.idx]):
            if p != 0:
                n.append(store[i])
        self.nbrs = n
        return n

    def __eq__(self, other):
        return self.idx == other.idx

    def __str__(self):
        return "{} (x: {}, y: {})".format(self.name, self.x, self.y)

    def __repr__(self):
        return str(self)


class PriceMap(object):
    """
    PriceMap encapsulates a 2D array that stores flight charges from City
    objects in A(rows) to B(columns).
    Use city.idx to access values.
    """
    def __init__(self, lines):
        self.pricemap = []
        for line in lines:
            strings = line.split()
            strings = [int(x) for x in strings]
            self.pricemap.append(strings)

    def price(self, a, b):
        return self.pricemap[a][b]

    def leastCost(self, store):
        # cheap hack
        if len(store) < 10:
            return 0

        m = sys.maxsize
        for i, ct in enumerate(store):
            for j in range(i+1, len(store)):
                if j < len(store):
                    fc = self.price(i, j)
                    if fc != 0:
                        x = store[j]
                        d = ct.distance(x)
                        ratio = fc/d
                        if ratio < m:
                            m = ratio
        return m

        def __str__(self):
            return "{}".format(self.pricemap)

    def __repr(self):
        return str(self)


class Info(object):
    """
    Info objects store global data required for CSE841 HW1.
    TODO: Make this a singleton.
    """
    def __init__(self, params, flights, cities):
        if len(params) == 4:
            self.startidx = params[0]
            self.endidx = params[1]
            self.hourly = params[2]
            self.future = params[3]

            # Initialize price map. This is used for neighbours and travelCost.
            l = getFileData(flights)
            self.pmap = PriceMap(l)

            # Read in City data
            self.store = []
            file = getFileData(cities)
            for i, city in enumerate(file):
                self.store.append(City(city, i))

            # Assign start and end cities
            self.start = self.store[self.startidx if self.startidx > 0 else 0]
            self.end = self.store[self.endidx if self.endidx < 60 else 59]

            # Populate each Node's nbrs property
            for n in self.store:
                n.neighbours(self.store, self.pmap)

            # Calculate least cost/mile
            self.least = self.pmap.leastCost(self.store)


class PQ(object):
    """
    A simple Priority Queue built on a min heap.
    Built on the idea from
    https://joernhees.de/blog/2010/07/19/min-heap-in-python/
    """
    def __init__(self, initial, key=lambda x: x):
        self.key = key
        if initial:
            self.data = [(key, item) for item in initial]
            heapq.heapify(self.data)
        else:
            self.data = []

    def peek(self):
        return self.data[0][1] if len(self.data) > 0 else 0

    def push(self, item):
        heapq.heappush(self.data, (self.key(item), item))

    def pop(self):
        return heapq.heappop(self.data)[1]

    def deprioritize(self, d, city):
        elements = [item for item in self.data if item[1] == city]
        if len(elements) > 0:
            i = self.data.index(elements[0])
            self.data[i] = (d, city)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "{}".format([x[1] for x in self.data])


class Flights(object):
    """
    Contains helper functions for calculting travel metrics between
    two City() objects, a and b.
    """
    def __init__(self, a, b, cruise=450):
        self.a = a
        self.b = b
        self.cruise = cruise

    def flightTime(self):
        d = self.a.distance(self.b)
        return (20.0/60)+(d/self.cruise)

    def waitTime(self):
        t = 0
        if not self.a.parent:
            return t
        if self.a.hub and self.b.hub:
            t = 1
            return t
        elif self.a.hub or self.b.hub:
            t = 2
        elif not self.a.hub and not self.b.hub:
                t = 3
        return t

    def totalTime(self):
        return self.flightTime() + self.waitTime()

    def travelCost(self, pm, h):
        """
        Returns Price[matrix] + hourly*(flight + wait)
        """
        p = pm.price(self.a.idx, self.b.idx)
        if p == 0:
            return 0
        time = self.totalTime()
        tcost = h * time
        # print(p,time,tcost)
        return p + tcost


# ------------------------------------------------------------------------------
# Dijkstra's Algorithm
def djk_distance(self, other, pm):
    return pm.price(self.idx, other.idx)


def djk(graph):
    distance = {}
    pq = PQ([], lambda x: distance[x.name])

    for city in graph.store:
        distance[city.name] = sys.maxsize
        city.parent = None
        city.known = False
        pq.push(city)

    # print(pq.data)
    pq.deprioritize(0, graph.start)
    distance[graph.start.name] = 0

    while len(pq) > 0:
        print(pq.data[0])
        n = pq.pop()
        n.known = True
        for nbr in n.nbrs:
            alt = distance[n.name] + djk_distance(n, nbr, graph.pmap)
            # print(nbr.name, alt, distance[nbr.name])
            if (
                not nbr.known and
                alt < distance[nbr.name] and
                graph.pmap.price(n.idx, nbr.idx) > 0
            ):
                # print("THE JUNGLE")
                distance[nbr.name] = alt
                pq.deprioritize(alt, nbr)
                nbr.parent = n

    return graph.end


# ------------------------------------------------------------------------------
# main A* algorithm
def fc(current, other, graph):
    g = gc(current, other, graph)
    h = hc(other, graph)
    if not g:
        return (None, None, h)
    return (g+h, g, h)


def gc(node, other, graph):
    t = Flights(node, other)
    returng = t.travelCost(graph.pmap, graph.hourly)
    return returng


def hc(node, graph):
    if node == graph.end:
        return 0

    t = Flights(node, graph.end)
    time = t.totalTime()
    tcost = graph.hourly * time
    d = node.distance(graph.end)
    return (d * graph.least)+tcost


def lowestf(cur, nodes, graph):
    minf = fc(cur, cur, graph)
    if not minf:
        minf = sys.maxsize
    ming = 0
    minn = cur
    for n in nodes:
        if n != cur:
            f = fc(cur, n, graph)
            if not f:
                continue
            if f[0] < minf:
                minf = f[0]
                ming = f[1]
                minn = n

    return (minn, ming, minf)


def pathfind(graph):
    openl = []
    closel = []
    current = graph.start

    openl.append(graph.start)
    while True:
        currtup = lowestf(current, openl, graph)
        current = currtup[0]
        currentf = currtup[2]
        openl.remove(current)
        closel.append(current)

        if current == graph.end:
            break

        for nb in current.nbrs:
            t = Flights(current, nb)
            if (
                nb in closel or
                not t.travelCost(graph.pmap, graph.hourly)
            ):
                continue
            if fc(current, nb, graph) < currentf or nb not in openl:
                nb.parent = current
                if nb not in openl:
                    openl.append(nb)

    return current


# ------------------------------------------------------------------------------
def timeformat(hours):
    """
    I'm not entirely sure why yet, but using datetime rounds up
    recurring float hours, so instead we'll use this to print time in the
    HH:MM format.
    """
    secs = hours * 3600.0
    h = int(secs / 3600.0)
    m = int((secs / 60.0) % 60.0)
    dt = datetime.time(h, m)

    return dt.strftime("%H:%M")



# Start by getting argument list from command line
_p = getArguments()

info = Info(_p, "flightCharges", "cities")

# Based on the input parameter "future_cost", decide between djk and a*.
n = pathfind(info) if info.future == 1 else djk(info)

path = []
while n is not None:
    f = n.parent
    path.append(n)
    n = n.parent
path = [x for x in reversed(path)]

rollg = 0
rollt = 0
prevt = 0
for i, n in enumerate(path):
    if i < len(path)-1:
        travel = Flights(n, path[i+1])
        pathval = travel.travelCost(info.pmap, info.hourly)
        g = gc(path[i+1], n, info) if info.future > 0 else pathval
        o = path[i+1]
        rollg += g
        rollt += travel.totalTime()
        print("{:18} {:18} {} - {} ${:.2f}".format(
                n.name.strip("*"),
                o.name.strip("*"), timeformat(prevt), timeformat(rollt), g
                ))
        prevt = rollt

print("Total: ${:.1f}".format(rollg))

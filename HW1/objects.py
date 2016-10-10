#!/usr/bin/python

import sys
import math
import heapq
import helpers as hlp


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

# ------------------------------------------------------------------------------
# Class Definitions


class City(Node):
    """
    City is used to traverse through the graph by using each nodes
    nbr property and neighbours().
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
        m = sys.maxsize
        for i, ct in enumerate(store):
            for j in range(i+1, len(store)):
                if j < len(store):
                    fc = self.price(i, j)
                    if fc != 0:
                        x = store[j]
                        d = ct.distance(x)
                        ratio = fc/d if d != 0 else 0
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
            l = hlp.getFileData(flights)
            self.pmap = PriceMap(l)

            # Read in City data
            self.store = []
            file = hlp.getFileData(cities)
            for i, city in enumerate(file):
                self.store.append(City(city, i))

            # Assign start and end cities
            l = len(self.store)
            self.start = self.store[self.startidx if self.startidx > 0 else 0]
            self.end = self.store[self.endidx if self.endidx < l else (l)]

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
            heapq.heapify(self.data)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "\n".join([str(x) for x in self.data])

    def __iter__(self):
        return iter([x[1] for x in self.data])


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
        if not self.a or not self.b:
            return sys.maxsize
        p = pm.price(self.a.idx, self.b.idx)
        if p == 0:
            return 0
        time = self.totalTime()
        tcost = h * time
        # print(p, time, tcost)
        return p + tcost

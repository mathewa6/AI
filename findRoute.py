#!/usr/bin/python

import sys
import math
import fileinput
from argparse import ArgumentParser as args

def getArguments():
    arguments = args(description="Finds the cheapest route between 2 locations. See CSE841 HW1 for spec..")
    arguments.add_argument("start",action="store",metavar="start_city",type=int,help="Starting city for the A* algorithm")
    arguments.add_argument("end",action="store",metavar="end_city",type=int,help="Ending city for the A* algorithm")
    arguments.add_argument("hourly",action="store",metavar="cost_of_hour",type=float,help="Hourly cost for the A* algorithm")
    arguments.add_argument("future",action="store",metavar="future_cost",type=int,help="Whether or not to use the A* algorithm")
    ip = args.parse_args(arguments)

    return (ip.start,ip.end,ip.hourly,ip.future)

def getFileData(filename):
    lines = []
    with open(filename) as f:
        for line in f:
            #print(line.strip().split())
            lines.append(line.strip())
    return lines

class Node(object):
    def __init__(self):
        self.name = "Unknown"
        self.parent = None
        self.nbrs = None

    def __repr__(self):
        return "{} >>> {}".format(self.name, self.parent)

class City(Node):
    def __init__(self, line, idx):
        super(City, self).__init__()
        (name, loc) = line.split()
        (x,y) = loc.strip("()").split(",")
        self.name = name
        self.x = int(x)
        self.y = int(y)
        self.hub = True if "*" in self.name else False
        self.idx = idx

    def distance(self, other):
        dx = other.x - self.x
        dy = other.y - self.y
        return 4*math.sqrt(math.pow(dx,2)+math.pow(dy,2))

    def flightTime(self, other):
        cruise = 450
        d = self.distance(other)
        return (20.0/60)+(d/cruise)

    def waitTime(self, other):
        t = 0
        if self.hub and other.hub:
            t = 1
            return t
        elif self.hub or other.hub:
            t = 2
        elif not self.hub and not other.hub:
                t =3
        return t

    def travelTime(self, other, pm, h):
        p = pm.price(self.idx, other.idx)
        time = self.flightTime(other) + self.waitTime(other)
        tcost = h * time
        print(p,time,tcost)
        return p + tcost

    def neighbours(self, store, pm):
        if self.nbrs:
            return self.nbrs

        n = []
        for i, p in enumerate(pm.pricemap[self.idx]):
            if p != 0:
                n.append(store[i])
        self.nbrs = n
        return n

    def __str__(self):
        return "{} (x: {}, y: {})".format(self.name, self.x, self.y)

    def __repr__(self):
        return str(self)

class PriceMap(object):
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
                    fc = self.price(i,j)
                    if fc != 0:
                        x = store[j]
                        d = ct.distance(x)
                        ratio = fc/d
                        if ratio  < m:
                            m = ratio
        return m

        def __str__(self):
            return "{}".format(self.pricemap)

    def __repr(self):
        return str(self)

#Start by getting argument list from command line
_params = getArguments()
if len(_params) == 4:
    _startidx = _params[0]
    _endidx = _params[1]
    _hourly = _params[2]
    _future = _params[3]

#Initialize price map. This is used for neighbours and travelTime.
l = getFileData("flightCharges")
_pmap = PriceMap(l)

#Read in City data
_store = []
file = getFileData("cities")
for i, city in enumerate(file):
    _store.append(City(city,i))

#Assign start and end cities
_start = _store[_startidx]
_end = _store[_endidx]

#Populate each Node's nbrs property
for n in _store:
    n.neighbours(_store, _pmap)

#Calculate least cost/mile
least = _pmap.leastCost(_store)

print(_start, _end, _pmap.price(_startidx,_endidx))
print(_start.distance(_end))
print(_start.flightTime(_end))
print(_start.waitTime(_end))
print(_start.travelTime(_end, _pmap, _hourly))
print(_start.nbrs)
print(_end.nbrs)
print(_pmap.leastCost(_store))

#Start the main algorithm


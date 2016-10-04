#!/usr/bin/python

import sys
import math
import fileinput
import sets
import heapq

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
        #self.g is ONLY per destination
        self.g = 0
        self.h = 0

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
        if not self.parent:
            return t
        if self.hub and other.hub:
            t = 1
            return t
        elif self.hub or other.hub:
            t = 2
        elif not self.hub and not other.hub:
                t =3
        return t

    def totalTime(self, other):
        return self.flightTime(other) + self.waitTime(other)

    def travelCost(self, other, pm, h):
        """
        Returns Price[matrix] + hourly*(flight + wait)
        """
        p = pm.price(self.idx, other.idx)
        if p == 0:
            return None
        time = self.totalTime(other)
        tcost = h * time
        #print(p,time,tcost)
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

    def __eq__(self,other):
        return self.idx == other.idx

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

class  Info(object):
    def __init__(self,params,flights,cities):
        if len(params) == 4:
            self.startidx = params[0]
            self.endidx = params[1]
            self.hourly = params[2]
            self.future = params[3]

            #Initialize price map. This is used for neighbours and travelCost.
            l = getFileData(flights)
            self.pmap = PriceMap(l)

            #Read in City data
            self.store = []
            file = getFileData(cities)
            for i, city in enumerate(file):
                self.store.append(City(city,i))

            #Assign start and end cities
            self.start = self.store[self.startidx if self.startidx > 0 else 0]
            self.end = self.store[self.endidx if self.endidx < 60 else 59]

            #Populate each Node's nbrs property
            for n in self.store:
                n.neighbours(self.store, self.pmap)

            #Calculate least cost/mile
            self.least = self.pmap.leastCost(self.store)

class PQ(object):
    def __init__(self,initial,key=lambda x:x):
        self.key = key
        if initial:
            self.data = [(key,item) for item in initial]
            heapq.heapify(self.data)
        else:
            self.data = []

    def peek(self):
        return self.data[0][1] if len(self.data)>0 else 0

    def push(self, item):
        heapq.heappush(self.data, (self.key(item),item))

    def pop(self):
        return heapq.heappop(self.data)[1]

    def deprioritize(self,d,city):
        elements = [item for item in self.data if item[1] == city]
        if len(elements) > 0:
            i = self.data.index(elements[0])
            self.data[i] = (d,city)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "{}".format([x[1] for x in self.data])

def djk(graph):
    pq = PQ([],lambda x:x.distance(graph.start))
    distance = {}
    prev = {}

    distance[graph.start.name] = 0

    for city in graph.store:
        if city != graph.start:
            distance[city.name] = sys.maxsize
            prev[city.name] = None
        pq.push(city)

    #print(pq.data)

    while len(pq) > 0:
        n = pq.pop()
        #print(n)
        for nbr in n.nbrs:
            alt = distance[n.name] + n.distance(nbr)
            #print(alt, distance[nbr.name])
            if alt < distance[nbr.name]:
                #print("THE JUNGLE")
                distance[nbr.name] = alt
                prev[nbr.name] = n
                pq.deprioritize(alt,nbr)

    return prev

#Start by getting argument list from command line
_p = getArguments()

info = Info(_p,"flightCharges", "cities")

#Start the main algorithm
#for n in info.store:
#    print(n.name)

ans = djk(info)
v = info.end
print(v)
while v != info.start:
    print(ans[v.name])
    v = ans[v.name]

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

class City(object):
    def __init__(self, line):
        (name, loc) = line.split()
        (x,y) = loc.strip("()").split(",")
        self.name = name
        self.x = int(x)
        self.y = int(y)
        self.hub = True if "*" in self.name else False

    def __str__(self):
        return "{} (x: {}, y: {})".format(self.name, self.x, self.y)

    def __repr__(self):
        return str(self)

class DistanceMap(object):
    def __init__(self, lines):
        self.dmap = []
        for line in lines:
            strings = line.split()
            strings = [int(x) for x in strings]
            self.dmap.append(strings)

    def distance(self, a, b):
        return self.dmap[a][b]

    def __str__(self):
        return "{}".format(self.dmap)

    def __repr(self):
        return str(self)

#Start by getting argument list from command line
all = getArguments()
print(all)

#Read in City data
file = getFileData("_city")
for i in file:
    print(City(i))

#Initialize Distance map
dist = getFileData("flightCharges")
l = dist[0]
print(len(dist),len(l.split()))
print(DistanceMap([l]))

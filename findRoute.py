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
            print(line.strip().split())
            lines.append(line.strip())
    return lines

class City(object):
    def __init__(self):
        pass

    def __string__(self):
        return "{} ({},{})".format(self.name, self.x, self.y)

    def __repr__(self):
        return str(self)

class DistanceMap(object):
    def __init__(self):
        pass

all = getArguments()
print(all)
file = getFileData("_city")
print(file)

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
import objects as obj
import helpers as hlp

# Start by getting argument list from command line
_p = hlp.getArguments()

info = obj.Info(_p, "flightCharges", "cities")

# Based on the input parameter "future_cost", decide between djk and a*.
n = hlp.pathfind(info) if info.future == 1 else hlp.djk(info)

path = []
while n is not None:
    path.append(n)
    n = n.parent
path = [x for x in reversed(path)]

rollg = 0
rollt = 0
prevt = 0
for i, n in enumerate(path):
    if i < len(path)-1:
        travel = obj.Flights(n, path[i+1])
        prevt += travel.waitTime()
        pathval = travel.travelCost(info.pmap, info.hourly)
        g = pathval
        o = path[i+1]
        rollg += g
        rollt += travel.totalTime()
        print("{:>6}. {:<15} - {:>15} {} {}  ${:.2f}".format(
                i + 1,
                n.name.strip("*"),
                o.name.strip("*"),
                hlp.timeformat(prevt), hlp.timeformat(rollt), g
                ))
        prevt = rollt

print("     {}: $ {:.2f}".format('Total Cost', rollg))

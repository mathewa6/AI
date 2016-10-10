#!/usr/bin/python

from __future__ import division

import sys
import datetime
import objects as obj

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


# ------------------------------------------------------------------------------
# Dijkstra's Algorithm
def djk_distance(self, other, pm):
    return pm.price(self.idx, other.idx)


def djk(graph):
    distance = {}
    pq = obj.PQ([], lambda x: distance[x.name])

    for city in graph.store:
        distance[city.name] = sys.maxsize
        city.parent = None
        city.known = False
        pq.push(city)

    pq.deprioritize(0, graph.start)
    distance[graph.start.name] = 0

    while len(pq) > 0:
        n = pq.pop()
        n.known = True
        for nbr in n.nbrs:
            alt = distance[n.name] + djk_distance(n, nbr, graph.pmap)
            if (
                not nbr.known and
                alt < distance[nbr.name] and
                graph.pmap.price(n.idx, nbr.idx) > 0
            ):
                distance[nbr.name] = alt
                pq.deprioritize(alt, nbr)
                nbr.parent = n

    return graph.end


# ------------------------------------------------------------------------------
# main A* algorithm
def hc(city, inf):
    dest = inf.end
    a = city.distance(dest)
    a = a * inf.least
    travels = obj.Flights(city, dest)
    t = travels.totalTime()
    b = inf.hourly * t

    return a + b


def f(city, inf):
    ta = obj.Flights(city.parent, city)
    return ta.travelCost(inf.pmap, inf.hourly) + hc(city, inf)


def pathfind(graph):
    distance = {}

    pq = obj.PQ([], lambda x: f(x, graph))

    for city in graph.store:
        distance[city.name] = sys.maxsize
        city.parent = None
        city.known = False
        pq.push(city)

    pq.deprioritize(0, graph.start)
    distance[graph.start.name] = 0

    while len(pq) > 0:
        n = pq.pop()
        n.known = True
        for nbr in n.nbrs:
            t = obj.Flights(n, nbr)
            alt = (
                distance[n.name] +  # Distance so far
                t.travelCost(graph.pmap, graph.hourly) +  # Cost from pmap +TW
                hc(nbr, graph)  # Heuristic from neighbour to graph.end
                )
            if (
                not nbr.known and
                alt < distance[nbr.name] + hc(nbr, graph) and
                graph.pmap.price(n.idx, nbr.idx) > 0
            ):
                distance[nbr.name] = (
                            distance[n.name] +
                            t.travelCost(graph.pmap, graph.hourly)
                            )
                pq.deprioritize(alt, nbr)
                nbr.parent = n

    return graph.end


# ------------------------------------------------------------------------------


def timeformat(hours):
    """
    I'm not entirely sure why... yet, but using datetime rounds up
    recurring float hours, so instead we'll use this to print time in the
    H:MM format.
    """
    secs = hours * 3600.0
    h = int(secs / 3600.0)
    m = int((secs / 60.0) % 60.0)
    dt = datetime.time(h, m)

    return dt.strftime("%-H:%M")

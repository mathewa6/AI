#!/usr/bin/python

import sys
import math
import fileinput
from argparse import ArgumentParser as args

arguments = args(description="Finds the cheapest route between 2 locations. See CSE841 HW1 for spec..")
#arguments.add_argument("inputs",action="store",metavar="n",type=int,nargs="+",help="start,end,hourly,future")
arguments.add_argument("start",action="store",metavar="start_city",type=int,help="Starting city for the A* algorithm")
arguments.add_argument("end",action="store",metavar="end_city",type=int,help="Ending city for the A* algorithm")
arguments.add_argument("hourly",action="store",metavar="cost_of_hour",type=float,help="Hourly cost for the A* algorithm")
arguments.add_argument("future",action="store",metavar="future_cost",type=int,help="Whether or not to use the A* algorithm")
ip = args.parse_args(arguments)

print("Hello world", ip.start, ip.end, ip.hourly,ip.future)


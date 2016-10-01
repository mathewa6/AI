#!/usr/bin/python

import sys
import math
import fileinput
from argparse import ArgumentParser as args

arguments = args(description="Finds the cheapest route between 2 locations. See CSE841 HW1 for spec..")
arguments.add_argument("inputs",action="store",metavar="n",type=int,nargs="+",help="start,end,hourly,future")

ip = args.parse_args(arguments)

print("Hello world", ip.inputs)


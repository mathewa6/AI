README: CSE841 HW1

This program uses Python and has been tested on arctic.cse.msu.edu using Python 2.7.9.

All code is included in the findRoute.py file. The file is also marked executable. The only requirement is that the "cities" and "flightCharges" files MUST be in the same directory as findRoute.py

Parameters to the algorithm are passed in as argments to the command line executable just like regular Linux scripts.
The *position* of arguments matter and are ordered as per the HW spec: start_index, end_index, hourly_cost and future_cost.

For example, type:
./findRoute.py 0 24 10 1
to run findRoute.y with start_index = 0, end_index = 24, hourly_cost = 10 and future_cost = 1.

README: CSE841 HW2

This is an implementation of a PCA Net.
Written by Adi Mathew.
10/13/16

This program requires Python 3, numpy AND matplotlib. Without either of those
there will be compilation errors.

To run the program simply run as a regular script with the HW2 spec.

For example,

./pcanet.py -l 1 -f 841Fall16/traininglist.txt -d net.db -o report.txt;2D
will run teh PCANet in training with 1 epoch.

./pcanet.py -f 841Fall16/testinglist.txt -d net.db -o report.txt
will run the PCANet wih testinglist.

OUTPUT:
The script will generate IMAGE files in a folder titled Output_YYMMDD_HHMMSS.
ALl required output for the HW will be generated in the same folder
as pcanet.py

The folders for training and test data sets must be in the same folder as well.
When running, folder names can ONLY be used for training and test lists.

DO NOT
generate database and reports in nested folders. This is to always keep reports
and database files from the most recent training.

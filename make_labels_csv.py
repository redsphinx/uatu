import os
from itertools import combinations
import sys

dirname = sys.argv[1]

# dirname = "/home/gabi/Documents/THALES/data/testing/dirs/"
# dirname = "/home/gabi/PycharmProjects/uatu/PRID2011"
list_of_directories = os.listdir(dirname)
list_of_directories.sort()
print list_of_directories

csv_file_name = dirname + "/pair_labels.csv"
csv_file = open(csv_file_name, "wr")

for comb in combinations(list_of_directories, 2):
    if comb[0].split("_")[1] == comb[1].split("_")[1]:
        csv_file.write(str(comb[0]) + "," + str(comb[1]) + ",1\n")
    else:
        csv_file.write(str(comb[0]) + "," + str(comb[1]) + ",0\n")

csv_file.close()

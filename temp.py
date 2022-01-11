import pandas as pd
import csv


def readInFFData(fname) -> list:
    with open(fname, "r") as read_obj:
        # pass the file object to DictReader() to get the DictReader object
        dict_reader = csv.DictReader(read_obj)
        # get a list of dictionaries from dct_reader
        tree_csv = list(dict_reader)
        # print list of dict i.e. rows

    treeList = list()

    for tree_i in range(len(tree_csv)):

        treeList.append(
            {
                "x": float(tree_csv[tree_i]["x"]),
                "y": float(tree_csv[tree_i]["y"]),
                "height": float(tree_csv[tree_i]["ht"]),
                "crownHeight": float(tree_csv[tree_i]["crown_len"]),
                "crownBaseHeight": float(tree_csv[tree_i]["crown_base_ht"]),
                "crownRadius": float(tree_csv[tree_i]["dia"]) / 2.0,
            }
        )
    return treeList


fname = "fastfuels/testcsv.csv"
print(readInFFData(fname))

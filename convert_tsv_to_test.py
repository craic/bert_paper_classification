# convert_test_to_tsv.py

# Copyright 2019  Robert Jones  jones@craic.com   Craic Computing LLC

# This software is made freely available under the terms of the MIT license


# convert a TSV train file to TSV test

# TSV input format is
#
# <id> <label> <arbitrary char> <text>

# TSV output format is
#
# <id> <text>

import os
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="path to TSV file")

  
    args = vars(ap.parse_args())
    file  = args["file"]

    print("{:s}\t{:s}".format("id", "text"))

    with open(file, 'r') as f:
        for line in f:
            fields = line.rstrip().split("\t")
            print("{:s}\t{:s}".format(fields[0], fields[3]))



if __name__ == "__main__":
      main()

# evaluate_new_data.py

# Copyright 2019  Robert Jones  jones@craic.com   Craic Computing LLC

# This software is made freely available under the terms of the MIT license

# Given a file of new data records (in test format) and a file of prediction results, output a list of IDs that are predicted to be positive


import os
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True, help="path to training format TSV file")
    ap.add_argument("--results", required=True, help="path to prediction results")

  
    args = vars(ap.parse_args())
    test_tsv_file  = args["tsv"]
    results_file = args["results"]

    ids = []
    results = []

    i = 0
    with open(test_tsv_file, 'r') as f:
        for line in f:
            if i == 0:
                i += 1 
                continue
            fields = line.rstrip().split("\t")
            ids.append(fields[0])
            
    with open(results_file, 'r') as f:
        for line in f:
            fields = line.rstrip().split("\t")
            neg_prob = float(fields[0])
            pos_prob = float(fields[1])
            result = 0
            if pos_prob > neg_prob:
                result = 1
            results.append(result)

    # merge the list of ids with the list of results - only output the positive hits
    for i in range(len(ids)):
        if results[i] == 1:
            print(ids[i])

            
if __name__ == "__main__":
      main()

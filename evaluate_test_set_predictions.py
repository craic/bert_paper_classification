# evaluate_test_set_predictions.py


# Copyright 2019  Robert Jones  jones@craic.com   Craic Computing LLC

# This software is made freely available under the terms of the MIT license

# given a test file and a file of prediction results, report the record ID, the label and the prediction

# TSV input format is
#
# <id> <label> <arbitrary char> <text>

# Results file format is
#
# <probability state 0> <probability state 1>

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
    labels = []
    results = []

    i = 0
    with open(test_tsv_file, 'r') as f:
        for line in f:
            if i == 0:
                i += 1 
                continue
            fields = line.rstrip().split("\t")
            ids.append(fields[0])
            label = 0
            if fields[1] == "1":
                label = 1
            labels.append(label)
            
    with open(results_file, 'r') as f:
        for line in f:
            fields = line.rstrip().split("\t")
            neg_prob = float(fields[0])
            pos_prob = float(fields[1])
            result = 0
            if pos_prob > neg_prob:
                result = 1
            results.append(result)

    n = len(ids)
    n_correct = 0
    
    for i in range(len(ids)):
        flag = ""
        if labels[i] == results[i]:
            flag = "*"
            n_correct += 1
        
        print("{:s}\t{:d}\t{:d}\t{:s}".format(ids[i], labels[i], results[i], flag))


    print("n  {:d}  correct  {:d}   {:3f}".format(n, n_correct, float(n_correct)/n))
            
if __name__ == "__main__":
      main()

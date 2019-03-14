# convert_yml_to_tsv.py

# Copyright 2019  Robert Jones  jones@craic.com   Craic Computing LLC

# This software is made freely available under the terms of the MIT license

# yaml format is
#
#- id: '28955949'
#  text: 'Anti-IL-17A blocking antibody reduces cyclosporin A-induced relapse in experimental
#    autoimmune encephalomyelitis mice. CNS, central nervous system CsA, cyclosporine

# TSV output format is
#
# <id> <label> <arbitrary char> <text>


import os
import yaml
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file",  required=True, help="path to yaml file")
    ap.add_argument("--label", required=True, help="label - integer")
  
    args = vars(ap.parse_args())
    file  = args["file"]
    label = args["label"]

    data = []
    with open(file, 'r') as f:
        data = yaml.load(f)

    static_char = 'a'
        
    for record in data:
        id   = record['id']
        text = record['text']

        # strip internal newlines and tabs
        text = text.replace('\n',' ')
        text = text.replace('\t',' ')
        
        # just get 1000 characters
        #text = text[:1000]
        # ... output the entire text - let bert select N words
        # convert the text to lower case - unlclear if bert does this

        text = text.lower()
        
        print("{:s}\t{:s}\t{:s}\t{:s}".format(id, label, static_char, text))



if __name__ == "__main__":
      main()

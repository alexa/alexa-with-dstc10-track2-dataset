from dataset_walker import DatasetWalker
from jsonschema import validate

import os
import sys
import argparse
import json
import warnings

def main(argv):
    parser = argparse.ArgumentParser(description='Check the validity of system outputs.')
    
    parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', choices=['train', 'val', 'test'], required=True, help='The dataset to analyze')
    parser.add_argument('--dataroot',dest='dataroot',action='store', metavar='PATH', required=True,
                        help='Will look for corpus in <dataroot>/<dataset>/...')

    parser.add_argument('--outfile',dest='outfile',action='store',metavar='JSON_FILE',required=True,
                        help='File containing output JSON')
    parser.add_argument('--schema',dest='schema_file',action='store',metavar='JSON_FILE',required=False,
                        default="output_schema.json", help='Output schema JSON file')

    args = parser.parse_args()
    
    data = DatasetWalker(dataset=args.dataset, dataroot=args.dataroot)

    with open(os.path.join(args.dataroot, args.schema_file), 'r') as f:
        schema = json.load(f)
    
    with open(args.outfile, 'r') as f:
        output = json.load(f)

    # initial syntax check with the schema
    validate(instance=output, schema=schema)

    # check the number of labels
    if len(data) != len(output):
        raise ValueError("the number of instances between ground truth and output does not match")

    print("Found no error, output file is valid.")

if __name__ =="__main__":
    main(sys.argv)        

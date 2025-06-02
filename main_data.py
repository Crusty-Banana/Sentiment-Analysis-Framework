import argparse
from helpers import (preprocess_VLSP)

def main(args):
    if args.action == 'preprocess_VLSP':
        preprocess_VLSP(input_path=args.input_path, 
                        output_path=args.output_path,
                        data=args.data, 
                        label=args.label)
    elif args.action == 'preprocess_Amazon':
        # TODO: code this
        pass
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Data Impacts")

    parser.add_argument('--action', type=str, default='split', choices=['preprocess_VLSP'], help='Action to perform: train or inference or validation')
    
    parser.add_argument('--input_path', type=str, default='data/VLSP/OG-train.csv', help='path to input data')
    parser.add_argument('--output_path', type=str, default='data/VLSP/PP-train.csv', help='path to output data')
    parser.add_argument('--data', type=str, default='texts')
    parser.add_argument('--label', type=str, default='labels')

    args = parser.parse_args()
    main(args)
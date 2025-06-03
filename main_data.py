import argparse
from helpers import (preprocess_VLSP, 
                     preprocess_Amazon,
                     split_csv,
                     combine_batches)

def main(args):
    if args.action == 'preprocess_VLSP':
        preprocess_VLSP(input_path=args.input_path, 
                        output_path=args.output_path,
                        data=args.data, 
                        label=args.label)
    elif args.action == 'preprocess_Amazon':
        preprocess_Amazon(input_path=args.input_path, 
                        output_path=args.output_path,
                        data=args.data, 
                        label=args.label)
    elif args.action == 'split':
        # Split for translation
        split_csv(input_path=args.input_path, 
                output_path=args.output_path)
    elif args.action == 'combine':
        # Combine for translation
        combine_batches(original_csv_path=args.original_csv_path,
                        input_path=args.input_path, 
                        output_path=args.output_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Data Impacts")

    parser.add_argument('--action', type=str, default='split', choices=['preprocess_VLSP', 'preprocess_Amazon', 'split', 'combine'], help='Action to perform')
    
    parser.add_argument('--input_path', type=str, default='data/VLSP/OG-train.csv', help='path to input data')
    parser.add_argument('--output_path', type=str, default='data/VLSP/PP-train.csv', help='path to output data')
    parser.add_argument('--data', type=str, default='texts')
    parser.add_argument('--label', type=str, default='labels')

    parser.add_argument('--original_csv_path', type=str, default='data/Amazon/OG-train.csv')

    args = parser.parse_args()
    main(args)
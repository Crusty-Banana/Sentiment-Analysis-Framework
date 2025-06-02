import argparse
from helpers import (split_csv, 
                     combine_batches, 
                     preprocess_Da, 
                     preprocess_Do, 
                     sample_data, 
                     concate_data,
                     balance_the_data)

def main(args):
    if args.action == 'split':
        split_csv(data_path="Experiment_data/Original_dataset/" + args.data_name, input_file=args.input_file, data=args.data, label=args.label)
    elif args.action == 'combine':
        combine_batches(original_df_path="Experiment_data/Original_dataset/" + args.data_name + "/SP-" + args.input_file, 
               translated_batch_path="Experiment_data/Original_dataset/" + args.data_name + "/Translated_batch", 
               file_count=4, 
               output_path="Experiment_data/Original_dataset/" + args.data_name + "/Translated-SP-" + args.input_file)
    elif args.action == 'preprocess_Da':
        preprocess_Da(data_path="Experiment_data/Original_dataset/" + args.data_name, input_file="Translated-SP-" + args.input_file)
    elif args.action == 'preprocess_Do':
        preprocess_Do(data_path="Experiment_data/Original_dataset/" + args.data_name, input_file=args.input_file, data=args.data, label=args.label)
    elif args.action == 'sample':
        sample_data(data_path=args.data_path, output_path=args.output_path, percent_sample_size=args.percent_sample_size)
    elif args.action == 'concate':
        concate_data(data_path1=args.data_path1, data_path2=args.data_path2, output_path=args.output_path)
    elif args.action == 'balance':
        balance_the_data(data_path=args.data_path, output_path=args.output_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Data Impacts")

    parser.add_argument('--action', type=str, default='split', choices=['split', 'combine', 'preprocess_Da', 'preprocess_Do', 'sample', 'concate', 'balance'], help='Action to perform: train or inference or validation')
    
    # Split & combine
    parser.add_argument('--data_name', type=str, default='Flipkart', choices=['AIVIVN_2019', 'Flipkart', 'UIT_VSFC'], help='name of dataset')
    parser.add_argument('--input_file', type=str, default='Dataset-SA.csv', help='name of the file')
    parser.add_argument('--data', type=str, default='Summary')
    parser.add_argument('--label', type=str, default='Sentiment')

    # Sample
    parser.add_argument('--data_path', type=str, default='Experiment_data/Original_dataset/AIVIVN_2019', help='Directory to load data')
    parser.add_argument('--output_path', type=str, default='Experiment_data/Sampled_dataset/AIVIVN_2019', help='Directory to save data')
    parser.add_argument('--percent_sample_size', type=int, default=50, help='% Number of samples. (from 1-100)')

    # Concate
    parser.add_argument('--data_path1', type=str, default='', help='Directory to load first data')
    parser.add_argument('--data_path2', type=str, default='', help='Directory to load second data')

    args = parser.parse_args()
    main(args)
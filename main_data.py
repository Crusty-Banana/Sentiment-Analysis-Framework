import argparse
from helpers import (preprocess_VLSP, 
                     preprocess_Amazon,
                     split_csv,
                     combine_batches,
                     under_sample_data,
                     sample_data,
                     concate_data,
                     remove_nan_rows)

def vietnamese_data_info():
    """Display information about Vietnamese datasets and preprocessing"""
    print("\n" + "="*60)
    print("VIETNAMESE SENTIMENT ANALYSIS DATASETS")
    print("="*60)
    print("\n1. VLSP Dataset:")
    print("   - Vietnamese Language and Speech Processing dataset")
    print("   - Contains Vietnamese sentiment analysis data")
    print("   - Labels: positive, negative, neutral")
    print("   - Use: preprocess_VLSP action")
    
    print("\n2. Amazon Dataset (Translated):")
    print("   - Amazon product reviews translated to Vietnamese")
    print("   - Can be used for cross-lingual training")
    print("   - Use: preprocess_Amazon action")
    
    print("\n3. Data Processing Options:")
    print("   - Preprocessing: Clean and format text data")
    print("   - Splitting: Divide data for batch processing")
    print("   - Combining: Merge processed batches")
    print("   - Undersampling: Balance class distribution")
    
    print("\n4. Vietnamese Models Available:")
    print("   - PhoBERT: First Vietnamese BERT model")
    print("   - ViDeBERTa: State-of-the-art Vietnamese DeBERTa")
    print("   - XLM-RoBERTa: Multilingual baseline")
    print("   - CafeBERT: Vietnamese-adapted XLM-RoBERTa")
    print("   - ViT5: Vietnamese T5 model")

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
    elif args.action == 'undersampling':
        under_sample_data(input_path=args.input_path, 
                        output_path=args.output_path)
    elif args.action == 'sample':
        sample_data(input_path=args.input_path,
                    output_path=args.output_path,
                    percent_sample_size=args.percent_sample_size)
    elif args.action == 'concate':
        concate_data(data_path1=args.input_path,
                    data_path2=args.input_path2,
                    output_path=args.output_path)
    elif args.action == 'remove_nan':
        remove_nan_rows(input_path=args.input_path)
    elif args.action == 'info':
        vietnamese_data_info()
    else:
        print("Invalid action. Use --help to see available options.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vietnamese Sentiment Analysis Data Processing")

    parser.add_argument('--action', type=str, default='info', 
                       choices=['preprocess_VLSP', 'preprocess_Amazon', 'split', 'combine', 'undersampling', 'sample', 'concate', 'remove_nan', 'info'],
                       help='Action to perform on Vietnamese data')
    
    parser.add_argument('--input_path', type=str, default='data/VLSP/OG-train.csv', help='path to input data')
    parser.add_argument('--output_path', type=str, default='data/VLSP/PP-train.csv', help='path to output data')
    parser.add_argument('--data', type=str, default='texts', help='Column name for text data')
    parser.add_argument('--label', type=str, default='labels', help='Column name for labels')

    parser.add_argument('--original_csv_path', type=str, default='data/Amazon/OG-train.csv', 
                       help='Path to original CSV for combining')
    parser.add_argument('--percent_sample_size', type=int, default='100', 
                       help='percentage of data to sample (0-100)')
    parser.add_argument('--input_path2', type=str, default='', help='second path to input data')
    
    args = parser.parse_args()
    main(args)
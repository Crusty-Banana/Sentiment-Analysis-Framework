import argparse
from train import train_model_with_dataset, inference_model_with_dataset
from models import CustomBERTModel

def main(args):
    if args.action == 'train':
        train_model_with_dataset(model_name=args.model_name, 
                                 data_path=args.data_path, 
                                 model_path=args.model_path, 
                                 checkpoint_path=args.checkpoint_path, 
                                 batch_size=args.batch_size,
                                 device=args.device)
        print("Training model {} on dataset {}".format(args.model_name, args.data_path))
    elif args.action == 'inference':
        model = CustomBERTModel(device=args.device)
        model.load_model(args.inference_model)

        prediction = model.predict(args.inference_text)

        print("Inference model {} on dataset {}:".format(args.model_name, args.data_path))
        print(f"Input: {args.inference_text}.\nOutput: {prediction}.")
    elif args.action == 'test':
        inference_model_with_dataset(model_name=args.model_name, 
                                     data_path=args.test_data_path, 
                                     model_path=args.test_model,
                                     batch_size=args.batch_size,
                                     device=args.device)
        print("Validation model {} on dataset {}".format(args.model_name, args.data_path))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Data Impacts")

    parser.add_argument('--action', type=str, default='train', choices=['train', 'inference', 'test', 'data'], help='Action to perform: train or inference or validation')
    
    # For Training
    parser.add_argument('--data_path', type=str, default='Experiment_data/Evaluation_dataset/AIVIVN.csv', help='Directory to load data')
    parser.add_argument('--model_name', type=str, default='mBERT', help='Name of the model')
    parser.add_argument('--model_path', type=str, default='', help='Directory to load trained model')
    parser.add_argument('--checkpoint_path', type=str, default='models/default_model', help='Directory to save trained model')

    # For Inference
    parser.add_argument('--inference_model', type=str, default="models/bert-base-multilingual-cased_AIVIVN_2019", help='Inference Text')
    parser.add_argument('--inference_text', type=str, default='sản phẩm bị lỗi', help='Inference Text')

    # For Testing
    parser.add_argument('--test_model', type=str, default="")
    parser.add_argument('--test_data_path', type=str, default="")

    # Details
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--batch_size', type=int, default="4")

    args = parser.parse_args()
    main(args)
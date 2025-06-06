import argparse
from train import train_model_with_dataset, inference_model_with_dataset
from models import CustomBERTModel

def main(args):
    if args.action == 'train':
        checkpoint_path = args.checkpoint_path + "_" + args.model_name \
                        + '_lr' + str(args.learning_rate) + '_bs' + str(args.batch_size) + '_epoch' + str(args.epoch) \
                        + '_scheduler' + str(args.use_scheduler) + '_nw' + str(args.num_workers)
        train_model_with_dataset(model_name=args.model_name, 
                                 data_path=args.data_path, 
                                 model_path=args.model_path, 
                                 checkpoint_path=checkpoint_path, 
                                 batch_size=args.batch_size,
                                 validation_size=args.validation_size,
                                 epoch=args.epoch,
                                 learning_rate=args.learning_rate,
                                 num_workers=args.num_workers,
                                 use_scheduler=args.use_scheduler,
                                 device=args.device)
        print("Training model {} on dataset {}".format(args.model_name, args.data_path))
    elif args.action == 'inference':
        model = CustomBERTModel(device=args.device)
        model.load_model(args.model_path)

        prediction = model.predict(args.inference_text)

        print("Inference model {} on dataset {}:".format(args.model_name, args.data_path))
        print(f"Input: {args.inference_text}.\nOutput: {prediction}.")
    elif args.action == 'test':
        inference_model_with_dataset(model_name=args.model_name, 
                                     data_path=args.data_path, 
                                     model_path=args.model_path,
                                     batch_size=args.batch_size,
                                     device=args.device)
        print("Validation model {} on dataset {}".format(args.model_name, args.data_path))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Data Impacts")

    parser.add_argument('--action', type=str, default='train', choices=['train', 'inference', 'test'], help='Action to perform: train or inference or validation')
    
    # For Training
    parser.add_argument('--data_path', type=str, default='', help='Directory to load data')
    parser.add_argument('--model_name', type=str, default='mBert', help='Name of the model')
    parser.add_argument('--model_path', type=str, default='', help='Directory to load trained model')
    parser.add_argument('--checkpoint_path', type=str, default='', help='Directory to save trained model')

    # For Inference
    parser.add_argument('--inference_text', type=str, default='sản phẩm bị lỗi', help='Inference Text')

    # Details
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--batch_size', type=int, default="128")
    parser.add_argument('--validation_size', type=float, default="0.05")
    parser.add_argument('--epoch', type=int, default="10")
    parser.add_argument('--learning_rate', type=float, default="3e-5")
    parser.add_argument('--use_scheduler', type=int, default="1")
    parser.add_argument('--num_workers', type=int, default="32")

    args = parser.parse_args()
    main(args)
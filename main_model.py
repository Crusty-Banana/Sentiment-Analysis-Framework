import argparse
from train import train_model_with_dataset, inference_model_with_dataset, compare_vietnamese_models, get_model

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
        model = get_model(args.model_name, device=args.device)
        
        if args.model_path:
            model.load_model(args.model_path)

        prediction = model.predict(args.inference_text)
        sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
        print("Inference model {} on text:".format(args.model_name))
        print(f"Input: {args.inference_text}")
        print(f"Output: {sentiment_map.get(prediction, prediction)}")
        
    elif args.action == 'test':
        inference_model_with_dataset(model_name=args.model_name, 
                                     data_path=args.data_path, 
                                     model_path=args.model_path,
                                     batch_size=args.batch_size,
                                     device=args.device)
        print("Validation model {} on dataset {}".format(args.model_name, args.data_path))
        
    elif args.action == 'compare':
        compare_vietnamese_models(data_path=args.data_path,
                                 batch_size=args.batch_size,
                                 device=args.device)
        print("Comparing Vietnamese models on dataset {}".format(args.data_path))
        
    else:
        print("Invalid action. Choose from: train, inference, test, compare")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vietnamese Sentiment Analysis Framework")

    parser.add_argument('--action', type=str, default='train', 
                       choices=['train', 'inference', 'test', 'compare'], 
                       help='Action to perform: train, inference, test, or compare models')
    
    # For Training
    parser.add_argument('--data_path', type=str, default='', help='Directory to load data')
    parser.add_argument('--model_name', type=str, default='mBert', 
                       choices=['mBert', 'phobert', 'videbertta', 'xlm_roberta', 'cafebert', 'vit5', 'mbert'],
                       help='Name of the Vietnamese model to use')
    parser.add_argument('--model_path', type=str, default='', help='Directory to load trained model')
    parser.add_argument('--checkpoint_path', type=str, default='', help='Directory to save trained model')

    # For Inference
    parser.add_argument('--inference_text', type=str, default='sản phẩm bị lỗi', 
                       help='Vietnamese text for inference')

    # Training Details
    parser.add_argument('--device', type=str, default="cpu", help='Device to run on (cuda:0, cuda:1, cpu)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training/inference')
    parser.add_argument('--validation_size', type=float, default=0.05, help='Validation split ratio')
    parser.add_argument('--epoch', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--use_scheduler', type=int, default=1, help='Use learning rate scheduler (1=True, 0=False)')
    parser.add_argument('--num_workers', type=int, default=32, help='Number of workers for data loading')

    args = parser.parse_args()
    main(args)
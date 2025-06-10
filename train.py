from models import (CustomBERTModel, 
                    PhoBERTModel, 
                    ViDeBERTaModel, 
                    XLMRobertaModel, 
                    CafeBERTModel, 
                    ViT5Model)
from datasets import CustomTextDataset
from torch.utils.data import DataLoader
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

def get_model(model_name, device="cpu"):
    """Get the appropriate model based on model name"""    
    print(f"Loading model: {model_name} on device: {device}")
    
    try:
        if model_name.lower() == "mbert":
            return CustomBERTModel(device=device)
        elif model_name.lower() == "phobert":
            return PhoBERTModel(device=device)
        elif model_name.lower() == "videbertta":
            return ViDeBERTaModel(device=device)
        elif model_name.lower() == "xlm_roberta":
            return XLMRobertaModel(device=device)
        elif model_name.lower() == "cafebert":
            return CafeBERTModel(device=device)
        elif model_name.lower() == "vit5":
            return ViT5Model(device=device)
        else:
            available_models = ["mBert", "phobert", "videbertta", "xlm_roberta", "cafebert", "vit5"]
            print(f"Unknown model: {model_name}. Available models: {available_models}")
            print("Defaulting to mBERT...")
            return CustomBERTModel(device=device)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        print("Falling back to mBERT...")
        return CustomBERTModel(device=device)

def train_model_with_dataset(model_name="mBert", 
                             data_path="", 
                             model_path="",
                             checkpoint_path="",
                             batch_size=4, 
                             validation_size=0.005,
                             epoch=10,
                             learning_rate=3e-5,
                             use_scheduler=True,
                             num_workers=4,
                             device="cpu"):
    """Train a model using a dataset.

    Args:
        model_name (string): Name of the model (mBert, phobert, videbertta, xlm_roberta, cafebert, vit5).
        data_path (string): Path to the dataset.
        model_path (string): Path to load the model.
        checkpoint_path (string): Path to save the checkpoint.
        batch_size (int): Batch size for training.
        validation_size (float): Validation split ratio.
        epoch (int): Number of training epochs.
        learning_rate (float): Learning rate for training.
        use_scheduler (bool): Whether to use learning rate scheduler.
        num_workers (int): Number of workers for data loading.
        device (string): Device to run the model on.
    """
    print(f"Training {model_name} model on Vietnamese sentiment analysis...")
    
    #---------------- Load Model ----------------#
    model = get_model(model_name, device)
    
    if model_path != "":
        print(f"Loading pre-trained model from {model_path}")
        model.load_model(model_path)
    
    #---------------- Load Data ----------------#
    print(f"Loading data from {data_path}")
    train_data = pd.read_csv(data_path)
    train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_data['data'].tolist(),
        train_data['label'].tolist(),
        test_size=validation_size,
        random_state=42
    )

    print(f"Training samples: {len(train_texts)}, Validation samples: {len(val_texts)}")

    train_dataset = CustomTextDataset(texts=train_texts, labels=train_labels, tokenizer=model.tokenizer)
    val_dataset = CustomTextDataset(texts=val_texts, labels=val_labels, tokenizer=model.tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    #---------------- Train Model ----------------#
    print(f"Starting training for {epoch} epochs...")
    model.train(train_dataloader, val_dataloader, epochs=epoch, learning_rate=learning_rate, use_scheduler=use_scheduler)

    #-------------- Simple Inference -------------#
    vietnamese_test_texts = [
        "Sản phẩm này rất tuyệt vời!",
        "Tôi không hài lòng với chất lượng.",
        "Bình thường, không có gì đặc biệt."
    ]
    
    print("\nTesting on Vietnamese samples:")
    for text in vietnamese_test_texts:
        prediction = model.predict(text)
        sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        print(f"Text: '{text}' -> Prediction: {sentiment_map.get(prediction, prediction)}")

    #---------------- Save Model -----------------#
    print(f"Saving model to {checkpoint_path}")
    model.save_model(checkpoint_path)
    print("Training completed successfully!")

def inference_model_with_dataset(model_name="mBert", 
                                 data_path="", 
                                 model_path="models/AIVIVN_2019_model",
                                 batch_size=4, 
                                 device="cpu"):
    """Evaluate a model using a dataset.

    Args:
        model_name (string): Name of the model (mBert, phobert, videbertta, xlm_roberta, cafebert, vit5).
        data_path (string): Path of the dataset.
        model_path (string): Path to load the model.
        batch_size (int): Batch size for evaluation.
        device (string): Device to run the model on.
    """
    print(f"Evaluating {model_name} model on Vietnamese sentiment analysis...")

    #---------------- Load Model ----------------#
    model = get_model(model_name, device)
    
    if model_path != "":
        print(f"Loading model from {model_path}")
        model.load_model(model_path)

    #---------------- Load Data -----------------#
    print(f"Loading test data from {data_path}")
    test_data = pd.read_csv(data_path)

    test_texts = test_data['data'].tolist()
    test_labels = test_data['label'].tolist()

    print(f"Test samples: {len(test_texts)}")

    test_dataset = CustomTextDataset(texts=test_texts, labels=test_labels, tokenizer=model.tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #---------------- Test Model ----------------#
    print("Starting evaluation...")
    model.evaluate(test_dataloader)
    print("Evaluation completed!")

def compare_vietnamese_models(data_path="", batch_size=4, device="cpu"):
    """Compare different Vietnamese models on the same dataset"""
    print("Comparing Vietnamese language models...")
    
    model_names = ["mBert", "phobert", "xlm_roberta"]  # Start with available models
    
    # Load test data
    test_data = pd.read_csv(data_path)
    test_texts = test_data['data'].tolist()[:100]  # Use subset for quick comparison
    test_labels = test_data['label'].tolist()[:100]
    
    results = {}
    
    for model_name in model_names:
        try:
            print(f"\nTesting {model_name}...")
            model = get_model(model_name, device)
            
            # Create dataset and dataloader
            test_dataset = CustomTextDataset(texts=test_texts, labels=test_labels, tokenizer=model.tokenizer)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            # Evaluate
            model.evaluate(test_dataloader)
            results[model_name] = "Completed"
            
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            results[model_name] = f"Error: {e}"
    
    print("\nComparison Results:")
    for model_name, result in results.items():
        print(f"{model_name}: {result}")


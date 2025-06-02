from models import CustomBERTModel
from datasets import CustomTextDataset
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from helpers import soft_preprocess_df

data_path = {
    "AIVIVN_2019": {
        "train": "data/AIVIVN_2019/train.csv",
        "test": "data/AIVIVN_2019/test.csv"
    },
    "Flipkart": {
        "train": "data/Flipkart/SP-Dataset.csv",
        "test": "data/Flipkart/SP-Dataset.csv"
    },
    "UIT-VSFC": {
        "train": "data/UIT-VSFC/train.csv",
        "test": "data/UIT-VSFC/test.csv"
    }
}

acronym = {
    "mBERT": "bert-base-multilingual-cased",
    "bert-base-multilingual-cased": "bert-base-multilingual-cased",
    "AIVIVN": "AIVIVN_2019",
}

def train_model_with_dataset(model_name="bert-base-multilingual-cased", 
                             data_path="", 
                             model_path="",
                             checkpoint_path="models/AIVIVN_2019_model",
                             batch_size=4, device="cpu"):
    """Train a model using a dataset.

    Args:
        model_name (string): Name of the model.
        data_name (string): Name of the dataset.
        model_path (string): Path to load the model.
        checkpoint_path (string): Path to save the checkpoint.
    """
    model_name = acronym[model_name]

    train_data = pd.read_csv(data_path)
    train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_data['data'].tolist(),
        train_data['label'].tolist(),
        test_size=0.005,
        random_state=42
    )

    model = CustomBERTModel(device=device)
    if (model_path != ""):
        model.load_model(model_path)
    
    train_dataset = CustomTextDataset(texts=train_texts, labels=train_labels, tokenizer=model.tokenizer)
    val_dataset = CustomTextDataset(texts=val_texts, labels=val_labels, tokenizer=model.tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.train(train_dataloader, val_dataloader, epochs=10, learning_rate=3e-5)

    new_text = "This is an amazing example."
    prediction = model.predict(new_text)
    print(f"Predicted label for '{new_text}': {prediction}")

    model.save_model(checkpoint_path)

def inference_model_with_dataset(model_name="bert-base-multilingual-cased", 
                                 data_path="", 
                                 model_path="models/AIVIVN_2019_model",
                                 batch_size=4, device="cpu"):
    """Train a model using a dataset.

    Args:
        model_name (string): Name of the model.
        data_path (string): Path of the dataset.
        model_path (string): Path to load the model.
    """

    model = CustomBERTModel(device=device)
    if (model_path != ""):
        model.load_model(model_path)

    test_data = pd.read_csv(data_path)

    test_texts = test_data['data'].tolist()
    test_labels = test_data['label'].tolist()

    test_dataset = CustomTextDataset(texts=test_texts, labels=test_labels, tokenizer=model.tokenizer)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.evaluate(test_dataloader)


from models import CustomBERTModel
from datasets import CustomTextDataset
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

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
        model_name (string): Name of the model.
        data_name (string): Name of the dataset.
        model_path (string): Path to load the model.
        checkpoint_path (string): Path to save the checkpoint.
    """
    #---------------- Load Model ----------------#
    model = None
    if model_name == "mBert":
        model = CustomBERTModel(device=device)
    
    if (model_path != ""):
        model.load_model(model_path)
    
    #---------------- Load Data ----------------#
    train_data = pd.read_csv(data_path)
    train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_data['data'].tolist(),
        train_data['label'].tolist(),
        test_size=validation_size,
        random_state=42
    )

    train_dataset = CustomTextDataset(texts=train_texts, labels=train_labels, tokenizer=model.tokenizer)
    val_dataset = CustomTextDataset(texts=val_texts, labels=val_labels, tokenizer=model.tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    #---------------- Train Model ----------------#
    model.train(train_dataloader, val_dataloader, epochs=epoch, learning_rate=learning_rate, use_scheduler=use_scheduler)

    #-------------- Simple Inference -------------#
    new_text = "This is an amazing example."
    prediction = model.predict(new_text)
    print(f"Predicted label for '{new_text}': {prediction}")

    #---------------- Save Model -----------------#
    model.save_model(checkpoint_path)

def inference_model_with_dataset(model_name="mBert", 
                                 data_path="", 
                                 model_path="models/AIVIVN_2019_model",
                                 batch_size=4, device="cpu"):
    """Train a model using a dataset.

    Args:
        model_name (string): Name of the model.
        data_path (string): Path of the dataset.
        model_path (string): Path to load the model.
    """

    #---------------- Load Model ----------------#
    model = None
    if (model_name == "mBert"):
        model = CustomBERTModel(device=device)
    if (model_path != ""):
        model.load_model(model_path)

    #---------------- Load Data -----------------#
    test_data = pd.read_csv(data_path)

    test_texts = test_data['data'].tolist()
    test_labels = test_data['label'].tolist()

    test_dataset = CustomTextDataset(texts=test_texts, labels=test_labels, tokenizer=model.tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #---------------- Test Model ----------------#
    model.evaluate(test_dataloader)


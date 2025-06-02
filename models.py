from transformers import (BertForSequenceClassification,
                          BertTokenizer,
                          BertModel,
                          AutoTokenizer,
                          AutoModel)
import torch
from tqdm import tqdm  # Import tqdm
import torch as _torch
from sklearn.metrics import confusion_matrix
import numpy as np
import torch.nn as _nn
import torch.nn.functional as _F

# Model Class
class CustomBERTModel:
    class CustomModel(_nn.Module):
        def __init__(self,
                     model_name: str,
                     out_channels: int):
            super().__init__()
            self.bert = BertModel.from_pretrained(model_name, output_hidden_states=True)
            self.fc1 = _nn.Linear(768*3, 500)  
            self.fc2 = _nn.Linear(500, 250) 
            self.fc3 = _nn.Linear(250, out_channels) 
            self.relu = _nn.ReLU()
            self.dropout = _nn.Dropout(0) 

        def forward(self, input_ids, attention_mask):
            # Pass inputs through BERT
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = torch.cat([state[:, 0, :] for state in outputs.hidden_states[-3:]], dim=1)
            
            # Pass through fully connected layers
            x = self.dropout(self.relu(self.fc1(cls_embeddings)))
            x = self.dropout(self.relu(self.fc2(x)))
            logits = self.fc3(x)
            return logits

    def __init__(self, model_name="google-bert/bert-base-multilingual-cased", num_labels=2, device="cuda:1"):
        """Initialize the BERT model for classification.

        Args:
            model_name (str): Pre-trained BERT model name.
            num_labels (int): Number of output labels.
        """
        self.model = self.CustomModel(model_name=model_name, out_channels=num_labels)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.device = device

    def train(self, train_dataloader, val_dataloader, epochs=10, learning_rate=1e-3):
        """Train the BERT model.

        Args:
            train_dataloader (DataLoader): DataLoader for training data.
            val_dataloader (DataLoader): DataLoader for validation data.
            epochs (int): Number of epochs.
            learning_rate (float): Learning rate for the optimizer.
        """
        self.model.to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            # Wrap the train_dataloader with tqdm to show progress
            for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{epochs}", leave=False):
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                logits = self.model(input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            scheduler.step()
            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}")

            self.evaluate(val_dataloader)   

    def evaluate(self, dataloader):
        """Evaluate the model on a validation dataset and print the confusion matrix.

        Args:
            dataloader (DataLoader): DataLoader for validation data.
        """

        self.model.to(self.device)
        self.model.eval()
        total, correct = 0, 0
        all_labels = []
        all_predictions = []

        # Wrap the validation dataloader with tqdm to show progress
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                logits = self.model(input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(logits, dim=-1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

        accuracy = correct / total
        print(f"Validation Accuracy: {accuracy:.4f}")

        # Compute and print confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        print("Confusion Matrix:")
        print(cm)

    def predict(self, text, max_length=128):
        """Perform inference on a single text.

        Args:
            dataloader (DataLoader): DataLoader for validation data.
        """
        self.model.eval()
        self.model.to(self.device)

        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(logits, dim=-1).item()

        return prediction
    
    def save_model(self, save_path):
        """Save the trained model to a file.

        Args:
            save_path (str): Path to save the model.
        """
        torch.save(self.model, save_path+".pth")

        print(f"Model saved to {save_path}")

    def load_model(self, load_path):
        """Load a model from a file.

        Args:
            load_path (str): Path to load the model from.
        """
        self.model = torch.load(load_path+".pth")
        print(f"Model loaded from {load_path}")

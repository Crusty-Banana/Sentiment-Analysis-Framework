from transformers import (BertForSequenceClassification,
                          BertTokenizer,
                          BertModel,
                          AutoTokenizer,
                          AutoModel,
                          AutoModelForSequenceClassification,
                          XLMRobertaForSequenceClassification,
                          XLMRobertaTokenizer,
                          T5ForConditionalGeneration,
                          T5Tokenizer)
import torch
from tqdm import tqdm  # Import tqdm
import torch as _torch
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    classification_report
)
import numpy as np
import torch.nn as _nn
import torch.nn.functional as _F
import os

def safe_load_model_and_tokenizer(model_class, tokenizer_class, model_name, **kwargs):
    """Safely load model and tokenizer with offline fallback"""
    try:
        # Try loading with internet connection first
        model = model_class.from_pretrained(model_name, **kwargs)
        tokenizer = tokenizer_class.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        print(f"Failed to load {model_name} online: {e}")
        
        # Try loading from local cache
        try:
            print(f"Attempting to load {model_name} from local cache...")
            model = model_class.from_pretrained(model_name, local_files_only=True, **kwargs)
            tokenizer = tokenizer_class.from_pretrained(model_name, local_files_only=True)
            print(f"Successfully loaded {model_name} from local cache")
            return model, tokenizer
        except Exception as cache_e:
            print(f"Failed to load {model_name} from cache: {cache_e}")
            raise Exception(f"Could not load {model_name} either online or from cache")

# Model Class
class CustomBERTModel:
    class CustomModel(_nn.Module):
        def __init__(self,
                     model_name: str,
                     out_channels: int):
            super().__init__()
            try:
                self.bert = BertModel.from_pretrained(model_name, output_hidden_states=True)
            except Exception as e:
                print(f"Failed to load BERT model online: {e}")
                try:
                    print(f"Attempting to load BERT model from local cache...")
                    self.bert = BertModel.from_pretrained(model_name, output_hidden_states=True, local_files_only=True)
                    print(f"Successfully loaded BERT model from local cache")
                except Exception as cache_e:
                    print(f"Failed to load BERT model from cache: {cache_e}")
                    raise Exception(f"Could not load BERT model {model_name} either online or from cache")
            
            self.fc1 = _nn.Linear(768*5, 512)  
            self.fc2 = _nn.Linear(512, 256) 
            self.fc3 = _nn.Linear(256, out_channels) 
            self.relu = _nn.ReLU()
            self.dropout = _nn.Dropout(0) 

        def forward(self, input_ids, attention_mask):
            # Pass inputs through BERT
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = torch.cat([state[:, 0, :] for state in outputs.hidden_states[-5:]], dim=1)
            
            # Pass through fully connected layers
            x = self.dropout(self.relu(self.fc1(cls_embeddings)))
            x = self.dropout(self.relu(self.fc2(x)))
            logits = self.fc3(x)
            return logits

    def __init__(self, model_name="google-bert/bert-base-multilingual-cased", num_labels=3, device="cpu"):
        """Initialize the BERT model for classification.

        Args:
            model_name (str): Pre-trained BERT model namƒe.
            num_labels (int): Number of output labels.
        """            
        try:
            self.model = self.CustomModel(model_name=model_name, out_channels=num_labels)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            try:
                print(f"Attempting to load {model_name} from local cache...")
                self.tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True)
                self.model = self.CustomModel(model_name=model_name, out_channels=num_labels)
                print(f"Successfully loaded {model_name} from local cache")
            except Exception as cache_e:
                print(f"Failed to load from cache: {cache_e}")
                raise Exception(f"Could not load {model_name} either online or from cache")
        
        self.device = device

    def train(self, train_dataloader, val_dataloader, epochs=10, learning_rate=1e-3, use_scheduler=True):
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

            if use_scheduler:
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

        all_labels = []
        all_predictions = []
        all_probs = []

        # Wrap the validation dataloader with tqdm to show progress
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                logits = self.model(input_ids, attention_mask=attention_mask)

                predictions = torch.argmax(logits, dim=-1)
                probs = _F.softmax(logits, dim=-1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probs = np.array(all_probs)

        accuracy = np.mean(all_predictions == all_labels)
        print(f"Validation Accuracy: {accuracy:.4f}")

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        report = classification_report(all_labels, all_predictions)
        cm = confusion_matrix(all_labels, all_predictions)
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        output = f"""
--- Evaluation Results ---
Validation Accuracy: {accuracy:.4f}
Weighted Precision: {precision:.4f}
Weighted Recall: {recall:.4f}
Weighted F1-Score: {f1:.4f}
AUC-ROC: {roc_auc if isinstance(roc_auc, float) else roc_auc:.4f}

Classification Report:
{report}

Confusion Matrix:
{cm}
--------------------------
"""
        return output

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
        torch.save(self.model.state_dict(), save_path+".pth")

        print(f"Model saved to {save_path}")

    def load_model(self, load_path):
        """Load a model from a file.

        Args:
            load_path (str): Path to load the model from.
        """
        self.model.load_state_dict(torch.load(load_path+".pth", weights_only=True))
        print(f"Model loaded from {load_path}")


# PhoBERT Model - Specialized for Vietnamese
class PhoBERTModel:
    def __init__(self, model_name="vinai/phobert-base-v2", num_labels=3, device="cpu"):
        try:
            self.model, self.tokenizer = safe_load_model_and_tokenizer(
                AutoModelForSequenceClassification, 
                AutoTokenizer, 
                model_name, 
                num_labels=num_labels
            )
        except Exception as e:
            print(f"Failed to load PhoBERT model: {e}")
            raise
        
        self.device = device

    def train(self, train_dataloader, val_dataloader, epochs=10, learning_rate=2e-5, use_scheduler=True):
        self.model.to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{epochs}", leave=False):
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if use_scheduler:
                scheduler.step()
                
            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}")

            self.evaluate(val_dataloader)

    def evaluate(self, dataloader):
        self.model.to(self.device)
        self.model.eval()

        all_labels = []
        all_predictions = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                predictions = torch.argmax(outputs.logits, dim=-1)
                probs = _F.softmax(outputs.logits, dim=-1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probs = np.array(all_probs)

        accuracy = np.mean(all_predictions == all_labels)
        print(f"Validation Accuracy: {accuracy:.4f}")

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        report = classification_report(all_labels, all_predictions)
        cm = confusion_matrix(all_labels, all_predictions)
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        output = f"""
--- Evaluation Results ---
Validation Accuracy: {accuracy:.4f}
Weighted Precision: {precision:.4f}
Weighted Recall: {recall:.4f}
Weighted F1-Score: {f1:.4f}
AUC-ROC: {roc_auc if isinstance(roc_auc, float) else roc_auc:.4f}

Classification Report:
{report}

Confusion Matrix:
{cm}
--------------------------
"""
        return output

    def predict(self, text, max_length=256):
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
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(outputs.logits, dim=-1).item()

        return prediction

    def save_model(self, save_path):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"PhoBERT model saved to {save_path}")

    def load_model(self, load_path):
        self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        print(f"PhoBERT model loaded from {load_path}")


# ViDeBERTa Model - State-of-the-art Vietnamese model
class ViDeBERTaModel:
    def __init__(self, model_name="Fsoft-AIC/videberta-base", num_labels=3, device="cpu"):
        try:
            self.model, self.tokenizer = safe_load_model_and_tokenizer(
                AutoModelForSequenceClassification, 
                AutoTokenizer, 
                model_name, 
                num_labels=num_labels
            )
        except Exception as e:
            print(f"Failed to load ViDeBERTa model: {e}")
            raise
            
        self.device = device

    def train(self, train_dataloader, val_dataloader, epochs=10, learning_rate=1e-5, use_scheduler=True):
        self.model.to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{epochs}", leave=False):
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if use_scheduler:
                scheduler.step()
                
            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}")

            self.evaluate(val_dataloader)

    def evaluate(self, dataloader):
        self.model.to(self.device)
        self.model.eval()

        all_labels = []
        all_predictions = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                predictions = torch.argmax(outputs.logits, dim=-1)
                probs = _F.softmax(outputs.logits, dim=-1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probs = np.array(all_probs)

        accuracy = np.mean(all_predictions == all_labels)
        print(f"Validation Accuracy: {accuracy:.4f}")

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        report = classification_report(all_labels, all_predictions)
        cm = confusion_matrix(all_labels, all_predictions)
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        output = f"""
--- Evaluation Results ---
Validation Accuracy: {accuracy:.4f}
Weighted Precision: {precision:.4f}
Weighted Recall: {recall:.4f}
Weighted F1-Score: {f1:.4f}
AUC-ROC: {roc_auc if isinstance(roc_auc, float) else roc_auc:.4f}

Classification Report:
{report}

Confusion Matrix:
{cm}
--------------------------
"""
        return output

    def predict(self, text, max_length=512):
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
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(outputs.logits, dim=-1).item()

        return prediction

    def save_model(self, save_path):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"ViDeBERTa model saved to {save_path}")

    def load_model(self, load_path):
        self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        print(f"ViDeBERTa model loaded from {load_path}")


# XLM-RoBERTa Model - Strong multilingual baseline
class XLMRobertaModel:
    def __init__(self, model_name="FacebookAI/xlm-roberta-base", num_labels=3, device="cpu"):            
        try:
            self.model, self.tokenizer = safe_load_model_and_tokenizer(
                XLMRobertaForSequenceClassification, 
                XLMRobertaTokenizer, 
                model_name, 
                num_labels=num_labels
            )
        except Exception as e:
            print(f"Failed to load XLM-RoBERTa model: {e}")
            raise
            
        self.device = device

    def train(self, train_dataloader, val_dataloader, epochs=10, learning_rate=2e-5, use_scheduler=True):
        self.model.to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{epochs}", leave=False):
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if use_scheduler:
                scheduler.step()
                
            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}")

            self.evaluate(val_dataloader)

    def evaluate(self, dataloader):
        self.model.to(self.device)
        self.model.eval()
        all_labels = []
        all_predictions = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                predictions = torch.argmax(outputs.logits, dim=-1)
                probs = _F.softmax(outputs.logits, dim=-1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probs = np.array(all_probs)

        accuracy = np.mean(all_predictions == all_labels)
        print(f"Validation Accuracy: {accuracy:.4f}")

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        report = classification_report(all_labels, all_predictions)
        cm = confusion_matrix(all_labels, all_predictions)
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        output = f"""
--- Evaluation Results ---
Validation Accuracy: {accuracy:.4f}
Weighted Precision: {precision:.4f}
Weighted Recall: {recall:.4f}
Weighted F1-Score: {f1:.4f}
AUC-ROC: {roc_auc if isinstance(roc_auc, float) else roc_auc:.4f}

Classification Report:
{report}

Confusion Matrix:
{cm}
--------------------------
"""
        return output

    def predict(self, text, max_length=512):
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
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(outputs.logits, dim=-1).item()

        return prediction

    def save_model(self, save_path):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"XLM-RoBERTa model saved to {save_path}")

    def load_model(self, load_path):
        self.model = XLMRobertaForSequenceClassification.from_pretrained(load_path)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(load_path)
        print(f"XLM-RoBERTa model loaded from {load_path}")


# CafeBERT Model - XLM-RoBERTa adapted for Vietnamese
class CafeBERTModel:
    def __init__(self, model_name="uitnlp/CafeBERT", num_labels=3, device="cpu"):
        try:
            self.model, self.tokenizer = safe_load_model_and_tokenizer(
                AutoModelForSequenceClassification, 
                AutoTokenizer, 
                model_name, 
                num_labels=num_labels
            )
        except Exception as e:
            print(f"Failed to load CafeBERT model: {e}")
            raise
            
        self.device = device

    def train(self, train_dataloader, val_dataloader, epochs=10, learning_rate=2e-5, use_scheduler=True):
        self.model.to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{epochs}", leave=False):
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if use_scheduler:
                scheduler.step()
                
            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}")

            self.evaluate(val_dataloader)

    def evaluate(self, dataloader):
        self.model.to(self.device)
        self.model.eval()
        all_labels = []
        all_predictions = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                predictions = torch.argmax(outputs.logits, dim=-1)
                probs = _F.softmax(outputs.logits, dim=-1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probs = np.array(all_probs)

        accuracy = np.mean(all_predictions == all_labels)
        print(f"Validation Accuracy: {accuracy:.4f}")

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        report = classification_report(all_labels, all_predictions)
        cm = confusion_matrix(all_labels, all_predictions)
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        output = f"""
--- Evaluation Results ---
Validation Accuracy: {accuracy:.4f}
Weighted Precision: {precision:.4f}
Weighted Recall: {recall:.4f}
Weighted F1-Score: {f1:.4f}
AUC-ROC: {roc_auc if isinstance(roc_auc, float) else roc_auc:.4f}

Classification Report:
{report}

Confusion Matrix:
{cm}
--------------------------
"""
        return output

    def predict(self, text, max_length=512):
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
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(outputs.logits, dim=-1).item()

        return prediction

    def save_model(self, save_path):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"CafeBERT model saved to {save_path}")

    def load_model(self, load_path):
        self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        print(f"CafeBERT model loaded from {load_path}")


# ViT5 Model - Vietnamese T5 for text-to-text tasks
class ViT5Model:
    def __init__(self, model_name="VietAI/vit5-base", num_labels=3, device="cpu"):
        try:
            self.base_model, self.tokenizer = safe_load_model_and_tokenizer(
                T5ForConditionalGeneration, 
                T5Tokenizer, 
                model_name
            )
        except Exception as e:
            print(f"Failed to load ViT5 model: {e}")
            raise
        
        # Add classification head
        self.classifier = _nn.Linear(self.base_model.config.d_model, num_labels)
        self.device = device
        self.num_labels = num_labels

    def train(self, train_dataloader, val_dataloader, epochs=10, learning_rate=1e-4, use_scheduler=True):
        self.base_model.to(self.device)
        self.classifier.to(self.device)
        
        optimizer = torch.optim.AdamW(
            list(self.base_model.parameters()) + list(self.classifier.parameters()), 
            lr=learning_rate
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.base_model.train()
            self.classifier.train()
            total_loss = 0
            
            for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{epochs}", leave=False):
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                # Use encoder outputs for classification
                encoder_outputs = self.base_model.encoder(
                    input_ids=input_ids, 
                    attention_mask=attention_mask
                )
                
                # Pool the encoder outputs (use mean of sequence)
                pooled_output = encoder_outputs.last_hidden_state.mean(dim=1)
                logits = self.classifier(pooled_output)
                
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if use_scheduler:
                scheduler.step()
                
            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}")

            self.evaluate(val_dataloader)

    def evaluate(self, dataloader):
        self.base_model.to(self.device)
        self.classifier.to(self.device)
        self.base_model.eval()
        self.classifier.eval()
        
        all_labels = []
        all_predictions = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                encoder_outputs = self.base_model.encoder(
                    input_ids=input_ids, 
                    attention_mask=attention_mask
                )
                
                pooled_output = encoder_outputs.last_hidden_state.mean(dim=1)
                logits = self.classifier(pooled_output)

                predictions = torch.argmax(logits, dim=-1)
                probs = _F.softmax(logits, dim=-1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probs = np.array(all_probs)

        accuracy = np.mean(all_predictions == all_labels)
        print(f"Validation Accuracy: {accuracy:.4f}")

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        report = classification_report(all_labels, all_predictions)
        cm = confusion_matrix(all_labels, all_predictions)
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        output = f"""
--- Evaluation Results ---
Validation Accuracy: {accuracy:.4f}
Weighted Precision: {precision:.4f}
Weighted Recall: {recall:.4f}
Weighted F1-Score: {f1:.4f}
AUC-ROC: {roc_auc if isinstance(roc_auc, float) else roc_auc:.4f}

Classification Report:
{report}

Confusion Matrix:
{cm}
--------------------------
"""
        return output

    def predict(self, text, max_length=512):
        self.base_model.eval()
        self.classifier.eval()
        self.base_model.to(self.device)
        self.classifier.to(self.device)

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
            encoder_outputs = self.base_model.encoder(
                input_ids=input_ids, 
                attention_mask=attention_mask
            )
            
            pooled_output = encoder_outputs.last_hidden_state.mean(dim=1)
            logits = self.classifier(pooled_output)
            prediction = torch.argmax(logits, dim=-1).item()

        return prediction

    def save_model(self, save_path):
        torch.save({
            'base_model': self.base_model.state_dict(),
            'classifier': self.classifier.state_dict(),
            'num_labels': self.num_labels
        }, save_path + ".pth")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path + "_tokenizer")
        print(f"ViT5 model saved to {save_path}")

    def load_model(self, load_path):
        checkpoint = torch.load(load_path + ".pth", weights_only=True)
        self.base_model.load_state_dict(checkpoint['base_model'])
        self.classifier.load_state_dict(checkpoint['classifier'])
        
        # Load tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(load_path + "_tokenizer")
        print(f"ViT5 model loaded from {load_path}")

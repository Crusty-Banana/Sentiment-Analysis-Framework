import torch
from torch.utils.data import Dataset

class CustomTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """Initialize the dataset.

        Args:
            texts (list of str): Input texts.
            labels (list of int): Corresponding labels for the texts.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer for BERT.
            max_length (int): Maximum sequence length for tokenized inputs.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        if (type(text) != str):
            print(text)
        # Tokenize the text
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Return input IDs, attention mask, and label
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }
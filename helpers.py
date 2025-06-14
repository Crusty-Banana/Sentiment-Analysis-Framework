import pandas as pd
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score
import json
import os
import re
import nltk
from nltk.corpus import stopwords

load_dotenv()

def preprocess_VLSP(input_path="data/VLSP/train.csv", 
                    output_path="data/VLSP/PP-train.csv",
                    data="texts", 
                    label="labels"):
    """
    Preprocesses text data from a CSV file for sentiment analysis,
    specifically tailored for VLSP-like datasets.

    The function performs the following steps:
    1. Reads the CSV file into a pandas DataFrame.
    2. Selects and renames the specified data and label columns to 'data' and 'label'.
    3. Maps string labels ('POS', 'NEU', 'NEG') to numerical values (0, 1, 2) in the 'label' column.
    4. Removes a predefined set of Vietnamese stop words from the 'data' column.
    5. Converts text in the 'data' column to lowercase.
    6. Removes URLs and HTML tags from the 'data' column.
    7. Removes characters not belonging to the Vietnamese alphabet (including accented characters),
       digits (0-9), basic punctuation (.,!?\'\"), or whitespace from the 'data' column.
    8. Removes extra whitespace from the 'data' column (multiple spaces to single, strips ends).
    9. Removes the Unicode replacement character '�' from the 'data' column.
    10. Strips leading/trailing double quotes from the 'data' column.
    11. Removes rows with missing or empty data in the 'data' column.
    12. Saves the preprocessed DataFrame to the specified output CSV file.

    Args:
        input_path (str): The file path to the CSV data.
        output_path (str): The file path to the resulting CSV data.
        data (str): The name of the column containing the text data.
        label (str): The name of the column containing the labels.

    Returns:
        pandas.DataFrame: A DataFrame with preprocessed 'data' and 'label' columns.
                          The 'data' column contains the cleaned and lemmatized text.
    """
    df = pd.read_csv(input_path)

    df = df[[data, label]].rename(columns={data: 'data', label: 'label'})

    df['label'] = df['label'].map({'POS': 0, 'NEU': 1, 'NEG': 2})

    df['label'] = df['label'].astype(int)
    df['data'] = df['data'].astype(str)
    # stop_words = set(['không', 'là', 'và', 'của', 'được', 'có', 'một', 'trong', 'để', 'cho', 'này', 'cũng', 'như', 'với'])
    # df['data'] = df['data'].apply(lambda data: ' '.join([word for word in data.split() if word not in stop_words]))

    df['data'] = df['data'] \
                .map(lambda data: data.lower()) \
                .map(lambda data: re.sub(r'http\S+|www\S+|https\S+', '', data, flags=re.MULTILINE)) \
                .map(lambda data: re.sub(r'<.*?>', '', data)) \
                .map(lambda data: re.sub(r'[^a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ0-9\s.,!?\'"]', '', data)) \
                .map(lambda data: re.sub(r'\s+', ' ', data).strip()) 
    
    df['data'] = df['data'].str.replace('﻿', '', regex=False)
    
    df['data'] = df['data'].str.strip('"')
    df = df[df['data'].notna() & (df['data'] != "")]

    df.to_csv(output_path, index=False)
    return df

def preprocess_Amazon(input_path="data/Amazon/OG-train.csv", 
                    output_path="data/Amazon/PP-train.csv",
                    data="reviewText", 
                    label="rating"):
    """
    """
    df = pd.read_csv(input_path)

    df = df[[data, label]].rename(columns={data: 'data', label: 'label'})

    df['label'] = df['label'].map({5.0: 0, 4.0: 0, 3.0: 1, 2.0: 2, 1.0: 2})

    df['label'] = df['label'].astype(int)
    df['data'] = df['data'].astype(str)

    # Remove stop words (english)
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    df['data'] = df['data'].apply(lambda data: ' '.join([word for word in data.split() if word not in stop_words]))

    df['data'] = df['data'] \
                .map(lambda data: data.lower()) \
                .map(lambda data: re.sub(r'http\S+|www\S+|https\S+', '', data, flags=re.MULTILINE)) \
                .map(lambda data: re.sub(r'<.*?>', '', data)) \
                .map(lambda data: re.sub(r'[^a-z0-9\s.,!?\'"]', '', data)) \
                .map(lambda data: re.sub(r'\s+', ' ', data).strip()) 
    
    df['data'] = df['data'].str.strip('"')
    df = df[df['data'].notna() & (df['data'] != "")]

    df.to_csv(output_path, index=False)
    return df

def translation_request(idx, data):
    return {
                "custom_id": f"review-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": f"Translate to Vietnamese, preserving original sentiment for sentiment analysis. Output only the translation."},
                        {"role": "user", "content": f"{data}"}
                    ]
                }
            }

def split_into_batches(df, output_path, batch_size=50000):
    file_count = 0
    line_count = 0
    
    os.makedirs(output_path, exist_ok=True) 
    batch_file = open(output_path + f"/batchinput_{file_count}.jsonl", "w")

    for idx, row in df.iterrows():
        json_line = translation_request(idx, row['data'])
        batch_file.write(json.dumps(json_line) + "\n")
        line_count += 1

        if line_count >= batch_size:
            batch_file.close()
            file_count += 1
            line_count = 0
            batch_file = open(output_path + f"/batchinput_{file_count}.jsonl", "w")

    batch_file.close()
    return file_count

def combine_batches(original_csv_path, input_path, output_path, file_count=50000):
    translations = {}
    for i in range(file_count + 1):
        if not os.path.exists(input_path + f"/batchoutput_{i}.jsonl"):
            break
        with open(input_path + f"/batchoutput_{i}.jsonl", "r") as file:
            for line in file:
                response = json.loads(line)
                custom_id = response["custom_id"]
                translated_review = response["response"]["body"]["choices"][0]["message"]["content"]
                translations[custom_id] = translated_review

    df = pd.read_csv(original_csv_path)
    df["data"] = df.index.map(
        lambda idx: translations.get(f"review-{idx}", None)
    )

    df = df[df['data'].notna() & (df['data'] != "")]
    df.to_csv(output_path, index=False)
    return df

def select_data_by_labels(df, labels=[0, 2]):
    df = df[df['labels'] in labels]
    df.loc[df['labels'] == 2, 'labels'] = 1
    return df


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

def split_csv(input_path,
              output_path):
    df = pd.read_csv(input_path)

    split_into_batches(df=df, 
                       output_path=output_path, 
                       batch_size=50000)

def sample_data(input_path="", 
                output_path="", 
                percent_sample_size=50):
    df = pd.read_csv(input_path, index_col=0)
    df = df.sample(frac=percent_sample_size/100)
    df.to_csv(output_path, index=False)
    return df

def concate_data(data_path1="", 
                 data_path2="", 
                 output_path=""):
    df1 = pd.read_csv(data_path1, index_col=0)
    df2 = pd.read_csv(data_path2, index_col=0)

    df = pd.concat([df1, df2])
    df.to_csv(output_path, index=False)
    return df

def balance_the_data(data_path, output_path):
    df = pd.read_csv(data_path, index_col=0)
    
    df_label_0 = df[df["label"] == 0]
    df_label_1 = df[df["label"] == 1]
    
    max_size = min(len(df_label_0), len(df_label_1))

    df_label_0 = df_label_0.sample(n=max_size)
    df_label_1 = df_label_1.sample(n=max_size)

    df = pd.concat([df_label_0, df_label_1])
    df = df.sample(frac=1)

    df.to_csv(output_path, index=False)

def under_sample_data(input_path, output_path):
    df = pd.read_csv(input_path, index_col=0)
    count_neg, count_neu, count_pos = df['label'].value_counts().sort_index()
    min_count = min(count_neg, count_neu, count_pos)

    df_neg_sampled = df[df['label'] == 0].sample(min_count, random_state=42)
    df_neu_sampled = df[df['label'] == 1].sample(min_count, random_state=42)
    df_pos_sampled = df[df['label'] == 2].sample(min_count, random_state=42)
    df = pd.concat([df_neg_sampled, df_neu_sampled, df_pos_sampled])

    df.to_csv(output_path, index=True)

def remove_nan_rows(input_path):
    df = pd.read_csv(input_path)
    df = df.dropna(subset=['data', 'label'])
    df = df[df['data'].notna() & (df['data'] != "")]
    df = df[df['label'].notna() & (df['label'] != "")]
    df.to_csv(input_path, index=False)
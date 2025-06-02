import pandas as pd
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score
import json
import os

load_dotenv()

def soft_preprocess_df(df, data="Summary", label="Sentiment"):
    df = df[[data, label]].rename(columns={data: 'data', label: 'label'})

    label_encoder_sentiment = LabelEncoder()

    df['label'] = label_encoder_sentiment.fit_transform(df['label'])
    return df

def translation_request(idx, data):
    return {
                "custom_id": f"review-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": f"Translate review into Vietnamese, dont add meaning"},
                        {"role": "user", "content": f"{data}"}
                    ]
                }
            }

def split_into_batches(df, output_path="Experiment_data/Original_dataset/Flipkart/To_be_translated_batch", batch_size=50000):
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

def combine_batches(original_df_path, translated_batch_path, file_count, output_path):
    translations = {}
    for i in range(file_count + 1):
        with open(translated_batch_path + f"/batch_{i}.jsonl", "r") as file:
            for line in file:
                response = json.loads(line)
                custom_id = response["custom_id"]
                translated_review = response["response"]["body"]["choices"][0]["message"]["content"]
                translations[custom_id] = translated_review

    df = pd.read_csv(original_df_path)
    df["Translated_Data"] = df.index.map(
        lambda idx: translations.get(f"review-{idx}", None)
    )

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

def split_csv(data_path="Experiment_data/Original_dataset/Flipkart",
              input_file="Dataset-SA.csv", 
              data="Summary", 
              label="Sentiment"):
    df = pd.read_csv(data_path+"/"+input_file)
    df = soft_preprocess_df(df, 
                            data=data, 
                            label=label)
    df.to_csv(data_path+"/SP-"+input_file)

    split_into_batches(df=df, 
                       output_path=data_path+"/To_be_translated_batch", 
                       batch_size=50000)
    
def preprocess_Da(data_path="Experiment_data/Original_dataset/Flipkart",
                  input_file="Translated-SP-Dataset-SA.csv"):
    df = pd.read_csv(data_path+"/"+input_file)

    # Select only positive and negative reviews
    # 0: Negative, 1: Positive
    df = df[df["label"].isin([0, 2])]
    df = soft_preprocess_df(df, data="Translated_Data", label="label")
    df['data'] = df['data'].str.strip('"')
    df = df[df['data'].notna() & (df['data'] != "")]
    df.to_csv(data_path+"/PP-"+input_file)

def preprocess_Do(data_path="Experiment_data/Original_dataset/AIVIVN_2019",
                  input_file="train.csv",
                  data="comment",
                  label="label"):
    df = pd.read_csv(data_path+"/"+input_file)
    df = soft_preprocess_df(df, data=data, label=label)
    df['data'] = df['data'].str.strip('"')
    df = df[df['data'].notna() & (df['data'] != "")]
    df.to_csv(data_path+"/PP-"+input_file)

def sample_data(data_path="", 
                output_path="", 
                percent_sample_size=50):
    df = pd.read_csv(data_path, index_col=0)
    df = df.sample(frac=percent_sample_size/100)
    df.to_csv(output_path)
    return df

def concate_data(data_path1="", 
                 data_path2="", 
                 output_path=""):
    df1 = pd.read_csv(data_path1, index_col=0)
    df2 = pd.read_csv(data_path2, index_col=0)

    df = pd.concat([df1, df2])
    df.to_csv(output_path)
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

    df.to_csv(output_path)
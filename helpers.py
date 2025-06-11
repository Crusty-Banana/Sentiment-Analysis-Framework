import pandas as pd
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score
import json
import os
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from openai import OpenAI

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

    df.to_csv(output_path)
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

    df.to_csv(output_path)
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
        with open(input_path + f"/batch_{i}.jsonl", "r") as file:
            for line in file:
                response = json.loads(line)
                custom_id = response["custom_id"]
                translated_review = response["response"]["body"]["choices"][0]["message"]["content"]
                translations[custom_id] = translated_review

    df = pd.read_csv(original_csv_path)
    df["data"] = df.index.map(
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

def split_csv(input_path,
              output_path):
    df = pd.read_csv(input_path)

    split_into_batches(df=df, 
                       output_path=output_path, 
                       batch_size=50000)

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

def under_sample_data(input_path, output_path):
    df = pd.read_csv(input_path, index_col=0)
    count_neg, count_neu, count_pos = df['label'].value_counts().sort_index()
    min_count = min(count_neg, count_neu, count_pos)

    df_neg_sampled = df[df['label'] == 0].sample(min_count, random_state=42)
    df_neu_sampled = df[df['label'] == 1].sample(min_count, random_state=42)
    df_pos_sampled = df[df['label'] == 2].sample(min_count, random_state=42)
    df = pd.concat([df_neg_sampled, df_neu_sampled, df_pos_sampled])

    df.to_csv(output_path)

def over_sample_data(input_path, output_path):
    """
    Apply Random Over Sampling to balance the dataset by duplicating minority class samples.
    
    Args:
        input_path (str): Path to input CSV file
        output_path (str): Path to output CSV file
    """
    df = pd.read_csv(input_path, index_col=0)
    count_neg, count_neu, count_pos = df['label'].value_counts().sort_index()
    max_count = max(count_neg, count_neu, count_pos)

    # Oversample each class to the maximum count
    df_neg = df[df['label'] == 0]
    df_neu = df[df['label'] == 1]
    df_pos = df[df['label'] == 2]
    
    # Random oversampling with replacement
    df_neg_oversampled = df_neg.sample(max_count, replace=True, random_state=42)
    df_neu_oversampled = df_neu.sample(max_count, replace=True, random_state=42)
    df_pos_oversampled = df_pos.sample(max_count, replace=True, random_state=42)
    
    df = pd.concat([df_neg_oversampled, df_neu_oversampled, df_pos_oversampled])
    df = df.sample(frac=1, random_state=42)  # Shuffle the data

    df.to_csv(output_path)
    print(f"Over-sampling completed. Original: {len(pd.read_csv(input_path, index_col=0))}, New: {len(df)}")

def apply_smote(input_path, output_path, text_vectorizer='tfidf', max_features=5000):
    """
    Apply SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset.
    
    Args:
        input_path (str): Path to input CSV file
        output_path (str): Path to output CSV file
        text_vectorizer (str): Type of vectorizer ('tfidf' or 'count')
        max_features (int): Maximum number of features for text vectorization
    """
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    
    df = pd.read_csv(input_path, index_col=0)
    
    # Vectorize text data for SMOTE (SMOTE requires numerical features)
    if text_vectorizer == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    else:
        vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
    
    X = vectorizer.fit_transform(df['data']).toarray()
    y = df['label'].values
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Create new dataframe with synthetic samples
    # Note: For text data, we can't directly convert back from vectorized form to meaningful text    # So we'll use the closest original samples to represent the synthetic ones
    original_texts = df['data'].values
    synthetic_df_list = []
    
    for i, (x_synthetic, y_synthetic) in enumerate(zip(X_resampled, y_resampled)):
        # Find the most similar original text for synthetic samples
        if i < len(df):
            # Original sample
            synthetic_df_list.append({
                'data': original_texts[i],
                'label': y_synthetic
            })
        else:
            # Synthetic sample - find most similar original text
            # Handle division by zero in similarity calculation
            norms_X = np.linalg.norm(X, axis=1)
            norm_synthetic = np.linalg.norm(x_synthetic)
            
            # Avoid division by zero
            if norm_synthetic == 0:
                most_similar_idx = 0
            else:
                similarities = np.dot(X, x_synthetic) / (norms_X * norm_synthetic + 1e-8)
                most_similar_idx = np.argmax(similarities)
            
            synthetic_df_list.append({
                'data': f"[SYNTHETIC] {original_texts[most_similar_idx]}",
                'label': y_synthetic
            })
    
    synthetic_df = pd.DataFrame(synthetic_df_list)
    synthetic_df = synthetic_df.sample(frac=1, random_state=42)  # Shuffle
    
    synthetic_df.to_csv(output_path)
    print(f"SMOTE completed. Original: {len(df)}, New: {len(synthetic_df)}")

def llm_few_shot_generation(input_path, output_path, target_samples_per_class=1000, api_key=None):
    """
    Generate synthetic samples using LLM few-shot learning for data augmentation.
    
    Args:
        input_path (str): Path to input CSV file
        output_path (str): Path to output CSV file
        target_samples_per_class (int): Target number of samples per class
        api_key (str): OpenAI API key (optional, will use env variable if not provided)
    """
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        
        df = pd.read_csv(input_path, index_col=0)
        
        # Get label counts
        label_counts = df['label'].value_counts().sort_index()
        print(f"Current label distribution: {dict(label_counts)}")
        
        # Labels mapping for Vietnamese sentiment
        label_map = {0: 'tích cực (positive)', 1: 'trung tính (neutral)', 2: 'tiêu cực (negative)'}
        
        augmented_samples = []
        
        for label in [0, 1, 2]:
            current_count = label_counts.get(label, 0)
            if current_count >= target_samples_per_class:
                print(f"Label {label} already has enough samples ({current_count})")
                continue
                
            samples_needed = target_samples_per_class - current_count
            print(f"Generating {samples_needed} samples for label {label} ({label_map[label]})")
            
            # Get few-shot examples
            class_samples = df[df['label'] == label]['data'].tolist()
            few_shot_examples = class_samples[:5]  # Use 5 examples
            
            # Create few-shot prompt
            examples_text = "\n".join([f"- {example}" for example in few_shot_examples])
            
            prompt = f"""Bạn là một chuyên gia phân tích cảm xúc tiếng Việt. Hãy tạo ra {min(samples_needed, 50)} câu phản hồi tiếng Việt có cảm xúc {label_map[label]}.

Dựa trên các ví dụ sau:
{examples_text}

Yêu cầu:
1. Các câu phải là tiếng Việt tự nhiên
2. Cảm xúc phải rõ ràng và nhất quán ({label_map[label]})
3. Độ dài từ 10-50 từ
4. Đa dạng về chủ đề (sản phẩm, dịch vụ, trải nghiệm)
5. Mỗi câu trên một dòng, không đánh số

Tạo {min(samples_needed, 50)} câu:"""

            try:
                # Generate batch of samples
                batches = (samples_needed + 49) // 50  # Round up division
                
                for batch in range(batches):
                    batch_size = min(50, samples_needed - batch * 50)
                    if batch_size <= 0:
                        break
                        
                    batch_prompt = prompt.replace(f"{min(samples_needed, 50)}", str(batch_size))
                    
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "Bạn là một chuyên gia tạo dữ liệu để huấn luyện mô hình phân tích cảm xúc tiếng Việt."},
                            {"role": "user", "content": batch_prompt}
                        ],
                        temperature=0.8,
                        max_tokens=2000
                    )
                    
                    generated_text = response.choices[0].message.content
                    generated_samples = [line.strip() for line in generated_text.split('\n') if line.strip() and not line.strip().startswith('-') and len(line.strip()) > 10]
                    
                    for sample in generated_samples[:batch_size]:
                        augmented_samples.append({
                            'data': f"[LLM_GEN] {sample}",
                            'label': label
                        })
                    
                    print(f"Generated batch {batch + 1}/{batches} for label {label}")
                    
            except Exception as e:
                print(f"Error generating samples for label {label}: {e}")
                continue
        
        # Combine original and augmented data
        augmented_df = pd.DataFrame(augmented_samples)
        final_df = pd.concat([df, augmented_df], ignore_index=True)
        final_df = final_df.sample(frac=1, random_state=42)  # Shuffle
        
        final_df.to_csv(output_path)
        print(f"LLM few-shot generation completed. Original: {len(df)}, Added: {len(augmented_samples)}, Final: {len(final_df)}")
        
        # Print final distribution
        final_counts = final_df['label'].value_counts().sort_index()
        print(f"Final label distribution: {dict(final_counts)}")
        
    except Exception as e:
        print(f"Error in LLM few-shot generation: {e}")
        print("Make sure you have set OPENAI_API_KEY in your .env file")

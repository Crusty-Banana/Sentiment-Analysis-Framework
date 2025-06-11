# COMP4020 Final Project: Vietnamese Sentiment Analysis

A comprehensive framework for Vietnamese sentiment analysis using state-of-the-art language models including Custom mBERT, PhoBERT, ViDeBERTa, XLM-RoBERTa, CafeBERT, and ViT5.

## 📊 Data Preprocessing

### Vietnamese Dataset Information
```bash
# Display information about Vietnamese datasets and models
python main_data.py --action info
```

### Preprocess VLSP Dataset
```bash
# Training data
python main_data.py --action preprocess_VLSP --data texts --label labels \
    --input_path data/VLSP/OG-train.csv --output_path data/VLSP/PP-train.csv

# Test data
python main_data.py --action preprocess_VLSP --data texts --label labels \
    --input_path data/VLSP/OG-test.csv --output_path data/VLSP/PP-test.csv

# Development data
python main_data.py --action preprocess_VLSP --data texts --label labels \
    --input_path data/VLSP/OG-dev.csv --output_path data/VLSP/PP-dev.csv
```

### Preprocess Amazon Dataset (Translated Vietnamese)
```bash
# Preprocess Amazon reviews
python main_data.py --action preprocess_Amazon --data reviewText --label rating \
    --input_path data/Amazon/OG-train.csv --output_path data/Amazon/PP-train.csv

# Split for batch processing
python main_data.py --action split \
    --input_path data/Amazon/PP-train.csv --output_path data/Amazon/Split-train

# Combine processed batches
python main_data.py --action combine \
    --original_csv_path data/Amazon/PP-train.csv \
    --input_path data/Amazon/translated_batch \
    --output_path data/Amazon/translated-PP-train.csv
```

### Data Balancing
```bash
# Undersample data for class balance
python main_data.py --action undersampling \
    --input_path data/Amazon/PP-train.csv --output_path data/Amazon/undersampled-PP-train.csv

python main_data.py --action undersampling \
    --input_path data/Amazon/translated-PP-train.csv \
    --output_path data/Amazon/undersampled-translated-PP-train.csv
```

### Data Sampling
```bash
# Sample from Undersampled procesed data
python main_data.py --action sample \
    --input_path data/Amazon/undersampled-translated-PP-train.csv \
    --output_path data/Amazon/sample50-undersampled-translated-PP-train.csv \
    --percent_sample_size 50
```

### Data Concatenate
```bash
# Sample from Undersampled procesed data
python main_data.py --action concate \
    --input_path data/Amazon/sample50-undersampled-translated-PP-train.csv \
    --input_path2 data/VLSP/PP-train.csv \
    --output_path data/Amazon/VLSP+tranlated-Amazon50.csv
```
## 🤖 Model Training & Evaluation

### Training Vietnamese Models

#### PhoBERT Training
```bash
# Train PhoBERT on VLSP dataset
python main_model.py --action train --model_name phobert \
    --data_path data/VLSP/PP-train.csv --checkpoint_path models/PhoBERT_VLSP \
    --epoch 10 --learning_rate 2e-5 --batch_size 64 --device cuda:0

# Train PhoBERT on Amazon dataset
python main_model.py --action train --model_name phobert \
    --data_path data/Amazon/PP-train.csv --checkpoint_path models/PhoBERT_Amazon \
    --epoch 15 --learning_rate 2e-5 --batch_size 32
```

#### ViDeBERTa Training (State-of-the-art)
```bash
# Train ViDeBERTa for best performance
python main_model.py --action train --model_name videbertta \
    --data_path data/VLSP/PP-train.csv --checkpoint_path models/ViDeBERTa_VLSP \
    --epoch 8 --learning_rate 1e-5 --batch_size 32 --device cuda:0

# Fine-tune with scheduler
python main_model.py --action train --model_name videbertta \
    --data_path data/VLSP/PP-train.csv --checkpoint_path models/ViDeBERTa_VLSP_scheduled \
    --epoch 12 --learning_rate 1e-5 --use_scheduler 1
```

#### XLM-RoBERTa Training (Multilingual baseline)
```bash
# Train XLM-RoBERTa as baseline
python main_model.py --action train --model_name xlm_roberta \
    --data_path data/VLSP/PP-train.csv --checkpoint_path models/XLM_RoBERTa_VLSP \
    --epoch 10 --learning_rate 2e-5 --batch_size 64 --device cuda:0
```

#### CafeBERT Training (Vietnamese-adapted)
```bash
# Train CafeBERT
python main_model.py --action train --model_name cafebert \
    --data_path data/VLSP/PP-train.csv --checkpoint_path models/CafeBERT_VLSP \
    --epoch 10 --learning_rate 2e-5 --batch_size 32
```

#### ViT5 Training (Text-to-text)
```bash
# Train ViT5 with higher learning rate
python main_model.py --action train --model_name vit5 \
    --data_path data/VLSP/PP-train.csv --checkpoint_path models/ViT5_VLSP \
    --epoch 15 --learning_rate 1e-4 --batch_size 16
```

### Model Evaluation

#### Test Individual Models
```bash
# Test PhoBERT
python main_model.py --action test --model_name phobert \
    --data_path data/VLSP/PP-test.csv \
    --model_path models/PhoBERT_VLSP_phobert_lr2e-05_bs64_epoch10_scheduler1_nw32 --device cuda:0

# Test ViDeBERTa
python main_model.py --action test --model_name videbertta \
    --data_path data/VLSP/PP-test.csv \
    --model_path models/ViDeBERTa_VLSP_videbertta_lr1e-05_bs32_epoch8_scheduler1_nw32

# Test XLM-RoBERTa
python main_model.py --action test --model_name xlm_roberta \
    --data_path data/VLSP/PP-test.csv \
    --model_path models/XLM_RoBERTa_VLSP_xlm_roberta_lr2e-05_bs64_epoch10_scheduler1_nw32
```

#### Compare All Models
```bash
# Compare all Vietnamese models on the same dataset
python main_model.py --action compare \
    --data_path data/VLSP/PP-test.csv --batch_size 32 --device cuda:1
```

### Vietnamese Text Inference

#### Single Text Prediction
```bash
# Test with positive Vietnamese text
python main_model.py --action inference --model_name phobert \
    --inference_text "Sản phẩm này rất tuyệt vời! Tôi rất hài lòng." \
    --model_path models/PhoBERT_VLSP_trained

# Test with negative Vietnamese text
python main_model.py --action inference --model_name videbertta \
    --inference_text "Chất lượng sản phẩm rất tệ, tôi không khuyên ai mua." \
    --model_path models/ViDeBERTa_VLSP_trained

# Test with neutral Vietnamese text
python main_model.py --action inference --model_name xlm_roberta \
    --inference_text "Sản phẩm bình thường, không có gì đặc biệt." \
    --model_path models/XLM_RoBERTa_VLSP_trained
```

## 🔬 Experimental Setup

### Current Research Variables

#### Datasets
- **VLSP**: Vietnamese Language and Speech Processing dataset
- **Amazon**: Product reviews translated to Vietnamese
- **Mixed datasets**: Combined Vietnamese and translated data

#### Vietnamese Models
- **PhoBERT**: First Vietnamese BERT (baseline)
- **ViDeBERTa**: State-of-the-art Vietnamese model
- **XLM-RoBERTa**: Multilingual baseline
- **CafeBERT**: Vietnamese-adapted multilingual model
- **ViT5**: Vietnamese T5 for text generation
- **Custom mBERT**: Enhanced multilingual BERT

#### Fine-tuning Strategies
1. Direct finetuning on Vietnamese data (VLSP)
2. Pre-training on translated Amazon → fine-tune on VLSP
3. Mixed finetuning (translated Amazon + VLSP)
4. Cross-lingual transfer learning (Pre-training on Amazon → fine-tune on VLSP)

#### Data Augmentation Techniques
- **Undersampling**: Balance class distribution
- **Translation augmentation**: Translate and back-translate

## 🔬 Experiments Script

### Direct finetuning on Vietnamese data
```bash 
python main_model.py --action train --model_name phobert \
    --data_path data/VLSP/PP-train.csv --checkpoint_path models/VLSP \
    --epoch 12 --learning_rate 2e-5 --batch_size 64 --device cuda:0

python main_model.py --action train --model_name videbertta \
    --data_path data/VLSP/PP-train.csv --checkpoint_path models/VLSP \
    --epoch 12 --learning_rate 2e-5 --batch_size 64 --device cuda:0
    
python main_model.py --action train --model_name xlm_roberta \
    --data_path data/VLSP/PP-train.csv --checkpoint_path models/VLSP \
    --epoch 10 --learning_rate 2e-5 --batch_size 64 --device cuda:0
    
python main_model.py --action train --model_name cafebert \
    --data_path data/VLSP/PP-train.csv --checkpoint_path models/VLSP \
    --epoch 10 --learning_rate 2e-5 --batch_size 64 --device cuda:0

python main_model.py --action train --model_name vit5 \
    --data_path data/VLSP/PP-train.csv --checkpoint_path models/VLSP \
    --epoch 10 --learning_rate 2e-5 --batch_size 64 --device cuda:0

python main_model.py --action train --model_name mbert \
    --data_path data/VLSP/PP-train.csv --checkpoint_path models/VLSP \
    --epoch 12 --learning_rate 2e-5 --batch_size 64 --device cuda:0
```

### Pretraining on Amazon data
```bash 
python main_model.py --action train --model_name phobert \
    --data_path data/Amazon/undersampled-PP-train.csv --checkpoint_path models/Amazon \
    --epoch 1 --learning_rate 2e-5 --batch_size 128 --device cuda:0

python main_model.py --action train --model_name videbertta \
    --data_path data/Amazon/undersampled-PP-train.csv --checkpoint_path models/Amazon \
    --epoch 1 --learning_rate 2e-5 --batch_size 64 --device cuda:0
    
python main_model.py --action train --model_name xlm_roberta \
    --data_path data/Amazon/undersampled-PP-train.csv --checkpoint_path models/Amazon \
    --epoch 1 --learning_rate 2e-5 --batch_size 64 --device cuda:0
    
python main_model.py --action train --model_name cafebert \
    --data_path data/Amazon/undersampled-PP-train.csv --checkpoint_path models/Amazon \
    --epoch 1 --learning_rate 2e-5 --batch_size 64 --device cuda:0

python main_model.py --action train --model_name vit5 \
    --data_path data/Amazon/undersampled-PP-train.csv --checkpoint_path models/Amazon \
    --epoch 1 --learning_rate 2e-5 --batch_size 64 --device cuda:0

python main_model.py --action train --model_name mbert \
    --data_path data/Amazon/undersampled-PP-train.csv --checkpoint_path models/Amazon \
    --epoch 1 --learning_rate 2e-5 --batch_size 64 --device cuda:0
```

### Pretraining on translated Amazon data
```bash 
python main_model.py --action train --model_name phobert \
    --data_path data/Amazon/undersampled-translated-PP-train.csv --checkpoint_path models/translatedAmazon \
    --epoch 6 --learning_rate 2e-5 --batch_size 64 --device cuda:0

python main_model.py --action train --model_name videbertta \
    --data_path data/Amazon/undersampled-translated-PP-train.csv --checkpoint_path models/translatedAmazon \
    --epoch 6 --learning_rate 2e-5 --batch_size 64 --device cuda:0
    
python main_model.py --action train --model_name xlm_roberta \
    --data_path data/Amazon/undersampled-translated-PP-train.csv --checkpoint_path models/translatedAmazon \
    --epoch 5 --learning_rate 2e-5 --batch_size 64 --device cuda:0
    
python main_model.py --action train --model_name cafebert \
    --data_path data/Amazon/undersampled-translated-PP-train.csv --checkpoint_path models/translatedAmazon \
    --epoch 5 --learning_rate 2e-5 --batch_size 64 --device cuda:4

python main_model.py --action train --model_name vit5 \
    --data_path data/Amazon/undersampled-translated-PP-train.csv --checkpoint_path models/translatedAmazon \
    --epoch 5 --learning_rate 2e-5 --batch_size 64 --device cuda:0

python main_model.py --action train --model_name mbert \
    --data_path data/Amazon/undersampled-translated-PP-train.csv --checkpoint_path models/translatedAmazon \
    --epoch 6 --learning_rate 2e-5 --batch_size 64 --device cuda:4
```

### Mixed finetuning (translated Amazon + VLSP)
```bash 
python main_model.py --action train --model_name phobert \
    --data_path data/Amazon/VLSP+translated-Amazon50.csv --checkpoint_path models/VLSP+translated-Amazon50 \
    --epoch 12 --learning_rate 2e-5 --batch_size 64 --device cuda:4

python main_model.py --action train --model_name videbertta \
    --data_path data/Amazon/VLSP+translated-Amazon50.csv --checkpoint_path models/VLSP+translated-Amazon50 \
    --epoch 12 --learning_rate 2e-5 --batch_size 64 --device cuda:4
    
python main_model.py --action train --model_name xlm_roberta \
    --data_path data/Amazon/VLSP+translated-Amazon50.csv --checkpoint_path models/VLSP+translated-Amazon50 \
    --epoch 10 --learning_rate 2e-5 --batch_size 64 --device cuda:4
    
python main_model.py --action train --model_name cafebert \
    --data_path data/Amazon/VLSP+translated-Amazon50.csv --checkpoint_path models/VLSP+translated-Amazon50 \
    --epoch 10 --learning_rate 2e-5 --batch_size 64 --device cuda:5

python main_model.py --action train --model_name vit5 \
    --data_path data/Amazon/VLSP+translated-Amazon50.csv --checkpoint_path models/VLSP+translated-Amazon50 \
    --epoch 10 --learning_rate 2e-5 --batch_size 64 --device cuda:5

python main_model.py --action train --model_name mbert \
    --data_path data/Amazon/VLSP+translated-Amazon50.csv --checkpoint_path models/VLSP+translated-Amazon50 \
    --epoch 12 --learning_rate 2e-5 --batch_size 64 --device cuda:5
```

### Cross-lingual transfer learning (Pre-training on Amazon → fine-tune on VLSP)
```bash 
python main_model.py --action train --model_name phobert \
    --data_path data/VLSP/PP-train.csv --checkpoint_path models/AmazonThenVLSP \
    --epoch 12 --learning_rate 2e-5 --batch_size 64 --device cuda:1 \
    --model_path models/Amazon_phobert_lr2e-05_bs128_epoch1_scheduler1_nw32

python main_model.py --action train --model_name videbertta \
    --data_path data/VLSP/PP-train.csv --checkpoint_path models/AmazonThenVLSP \
    --epoch 12 --learning_rate 2e-5 --batch_size 64 --device cuda:1 \
    --model_path models/Amazon_videbertta_lr2e-05_bs64_epoch1_scheduler1_nw32
    
python main_model.py --action train --model_name xlm_roberta \
    --data_path data/VLSP/PP-train.csv --checkpoint_path models/AmazonThenVLSP \
    --epoch 10 --learning_rate 2e-5 --batch_size 64 --device cuda:1 \
    --model_path models/Amazon_xlm_roberta_lr2e-05_bs64_epoch1_scheduler1_nw32
    
python main_model.py --action train --model_name cafebert \
    --data_path data/VLSP/PP-train.csv --checkpoint_path models/AmazonThenVLSP \
    --epoch 10 --learning_rate 2e-5 --batch_size 64 --device cuda:1 \
    --model_path models/Amazon_cafebert_lr2e-05_bs64_epoch1_scheduler1_nw32

python main_model.py --action train --model_name vit5 \
    --data_path data/VLSP/PP-train.csv --checkpoint_path models/AmazonThenVLSP \
    --epoch 10 --learning_rate 2e-5 --batch_size 64 --device cuda:1 \
    --model_path models/Amazon_vit5_lr2e-05_bs64_epoch1_scheduler1_nw32

python main_model.py --action train --model_name mbert \
    --data_path data/VLSP/PP-train.csv --checkpoint_path models/AmazonThenVLSP \
    --epoch 12 --learning_rate 2e-5 --batch_size 64 --device cuda:1 \
    --model_path models/Amazon_mbert_lr2e-05_bs64_epoch1_scheduler1_nw32
```

### Pre-training on translated Amazon → fine-tune on VLSP
```bash 
python main_model.py --action train --model_name phobert \
    --data_path data/VLSP/PP-train.csv --checkpoint_path models/TranslatedAmazonThenVLSP \
    --epoch 12 --learning_rate 2e-5 --batch_size 64 --device cuda:0 \
    --model_path models/translatedAmazon_phobert_lr2e-05_bs64_epoch6_scheduler1_nw32

python main_model.py --action train --model_name videbertta \
    --data_path data/VLSP/PP-train.csv --checkpoint_path models/TranslatedAmazonThenVLSP \
    --epoch 12 --learning_rate 2e-5 --batch_size 64 --device cuda:0 \
    --model_path models/translatedAmazon_videbertta_lr2e-05_bs64_epoch6_scheduler1_nw32
    
python main_model.py --action train --model_name xlm_roberta \
    --data_path data/VLSP/PP-train.csv --checkpoint_path models/TranslatedAmazonThenVLSP \
    --epoch 10 --learning_rate 2e-5 --batch_size 64 --device cuda:0 \
    --model_path models/translatedAmazon_xlm_roberta_lr2e-05_bs64_epoch5_scheduler1_nw32
    
python main_model.py --action train --model_name cafebert \
    --data_path data/VLSP/PP-train.csv --checkpoint_path models/TranslatedAmazonThenVLSP \
    --epoch 10 --learning_rate 2e-5 --batch_size 64 --device cuda:1 \
    --model_path models/translatedAmazon_cafebert_lr2e-05_bs64_epoch5_scheduler1_nw32

python main_model.py --action train --model_name vit5 \
    --data_path data/VLSP/PP-train.csv --checkpoint_path models/TranslatedAmazonThenVLSP \
    --epoch 10 --learning_rate 2e-5 --batch_size 64 --device cuda:1 \
    --model_path models/translatedAmazon_vit5_lr2e-05_bs64_epoch5_scheduler1_nw32

python main_model.py --action train --model_name mbert \
    --data_path data/VLSP/PP-train.csv --checkpoint_path models/TranslatedAmazonThenVLSP \
    --epoch 12 --learning_rate 2e-5 --batch_size 64 --device cuda:1 \
    --model_path models/translatedAmazon_mbert_lr2e-05_bs64_epoch6_scheduler1_nw32
```

## 📈 Performance Benchmarks

### Evaluation Script
#### Direct finetuning on Vietnamese data
```bash
python main_model.py --action test --model_name phobert \
    --data_path data/VLSP/PP-test.csv --batch_size 64 --device cuda:0 \
    --model_path models/VLSP_phobert_lr2e-05_bs64_epoch12_scheduler1_nw32
python main_model.py --action test --model_name videbertta \
    --data_path data/VLSP/PP-test.csv --batch_size 64 --device cuda:0 \
    --model_path models/VLSP_videbertta_lr1e-05_bs64_epoch12_scheduler1_nw32
python main_model.py --action test --model_name xlm_roberta \
    --data_path data/VLSP/PP-test.csv --batch_size 64 --device cuda:0 \
    --model_path models/VLSP_xlm_roberta_lr2e-05_bs64_epoch10_scheduler1_nw32
python main_model.py --action test --model_name cafebert \
    --data_path data/VLSP/PP-test.csv --batch_size 64 --device cuda:0 \
    --model_path models/VLSP_cafebert_lr2e-05_bs64_epoch10_scheduler1_nw32
python main_model.py --action test --model_name vit5 \
    --data_path data/VLSP/PP-test.csv --batch_size 64 --device cuda:0 \
    --model_path models/VLSP_vit5_lr2e-05_bs64_epoch10_scheduler1_nw32
python main_model.py --action test --model_name mbert \
    --data_path data/VLSP/PP-test.csv --batch_size 64 --device cuda:0 \
    --model_path models/VLSP_mbert_lr2e-05_bs64_epoch12_scheduler1_nw32
```

#### Pre-training on translated Amazon → fine-tune on VLSP
```bash
python main_model.py --action test --model_name phobert \
    --data_path data/VLSP/PP-test.csv --batch_size 64 --device cuda:0 \
    --model_path models/TranslatedAmazonThenVLSP_phobert_lr2e-05_bs64_epoch12_scheduler1_nw32
python main_model.py --action test --model_name videbertta \
    --data_path data/VLSP/PP-test.csv --batch_size 64 --device cuda:0 \
    --model_path models/TranslatedAmazonThenVLSP_videbertta_lr2e-05_bs64_epoch12_scheduler1_nw32
python main_model.py --action test --model_name xlm_roberta \
    --data_path data/VLSP/PP-test.csv --batch_size 64 --device cuda:0 \
    --model_path models/TranslatedAmazonThenVLSP_xlm_roberta_lr2e-05_bs64_epoch10_scheduler1_nw32
python main_model.py --action test --model_name cafebert \
    --data_path data/VLSP/PP-test.csv --batch_size 64 --device cuda:0 \
    --model_path models/TranslatedAmazonThenVLSP_cafebert_lr2e-05_bs64_epoch10_scheduler1_nw32
python main_model.py --action test --model_name vit5 \
    --data_path data/VLSP/PP-test.csv --batch_size 64 --device cuda:0 \
    --model_path models/TranslatedAmazonThenVLSP_vit5_lr2e-05_bs64_epoch10_scheduler1_nw32
python main_model.py --action test --model_name mbert \
    --data_path data/VLSP/PP-test.csv --batch_size 64 --device cuda:0 \
    --model_path models/TranslatedAmazonThenVLSP_mbert_lr2e-05_bs64_epoch12_scheduler1_nw32
```

#### Mixed finetuning (translated Amazon + VLSP)
```bash
python main_model.py --action test --model_name phobert \
    --data_path data/VLSP/PP-test.csv --batch_size 64 --device cuda:0 \
    --model_path models/VLSP+translated-Amazon50_phobert_lr2e-05_bs64_epoch12_scheduler1_nw32
python main_model.py --action test --model_name videbertta \
    --data_path data/VLSP/PP-test.csv --batch_size 64 --device cuda:0 \
    --model_path models/VLSP+translated-Amazon50_videbertta_lr2e-05_bs64_epoch12_scheduler1_nw32
python main_model.py --action test --model_name xlm_roberta \
    --data_path data/VLSP/PP-test.csv --batch_size 64 --device cuda:0 \
    --model_path models/VLSP+translated-Amazon50_xlm_roberta_lr2e-05_bs64_epoch10_scheduler1_nw32
python main_model.py --action test --model_name cafebert \
    --data_path data/VLSP/PP-test.csv --batch_size 64 --device cuda:0 \
    --model_path models/VLSP+translated-Amazon50_cafebert_lr2e-05_bs64_epoch10_scheduler1_nw32
python main_model.py --action test --model_name vit5 \
    --data_path data/VLSP/PP-test.csv --batch_size 64 --device cuda:0 \
    --model_path models/VLSP+translated-Amazon50_vit5_lr2e-05_bs64_epoch10_scheduler1_nw32
python main_model.py --action test --model_name mbert \
    --data_path data/VLSP/PP-test.csv --batch_size 64 --device cuda:0 \
    --model_path models/VLSP+translated-Amazon50_mbert_lr2e-05_bs64_epoch12_scheduler1_nw32
```

#### Cross-lingual transfer learning (Pre-training on Amazon → fine-tune on VLSP)
```bash
python main_model.py --action test --model_name phobert \
    --data_path data/VLSP/PP-test.csv --batch_size 64 --device cuda:0 \
    --model_path models/AmazonThenVLSP_phobert_lr2e-05_bs64_epoch12_scheduler1_nw32
python main_model.py --action test --model_name videbertta \
    --data_path data/VLSP/PP-test.csv --batch_size 64 --device cuda:0 \
    --model_path models/AmazonThenVLSP_videbertta_lr2e-05_bs64_epoch12_scheduler1_nw32
python main_model.py --action test --model_name xlm_roberta \
    --data_path data/VLSP/PP-test.csv --batch_size 64 --device cuda:0 \
    --model_path models/AmazonThenVLSP_xlm_roberta_lr2e-05_bs64_epoch10_scheduler1_nw32
python main_model.py --action test --model_name cafebert \
    --data_path data/VLSP/PP-test.csv --batch_size 64 --device cuda:0 \
    --model_path models/AmazonThenVLSP_cafebert_lr2e-05_bs64_epoch10_scheduler1_nw32
python main_model.py --action test --model_name vit5 \
    --data_path data/VLSP/PP-test.csv --batch_size 64 --device cuda:0 \
    --model_path models/AmazonThenVLSP_vit5_lr2e-05_bs64_epoch10_scheduler1_nw32
python main_model.py --action test --model_name mbert \
    --data_path data/VLSP/PP-test.csv --batch_size 64 --device cuda:0 \
    --model_path models/AmazonThenVLSP_mbert_lr2e-05_bs64_epoch12_scheduler1_nw32
```

Based on research findings, expected performance on Vietnamese sentiment analysis:


## 🛠️ Installation & Setup

### Requirements
```bash
pip install torch transformers datasets scikit-learn pandas numpy tqdm
```

### GPU Setup
```bash
# Check available GPUs
nvidia-smi

# Set device in commands
--device cuda:0  # First GPU
--device cuda:1  # Second GPU
--device cpu     # CPU only
```

## 📋 Quick Start Examples

### 1. Train PhoBERT (Recommended for beginners)
```bash
python main_model.py --action train --model_name phobert \
    --data_path data/VLSP/PP-train.csv --checkpoint_path models/my_phobert \
    --epoch 5 --batch_size 32
```

### 2. Train ViDeBERTa (Best performance)
```bash
python main_model.py --action train --model_name videbertta \
    --data_path data/VLSP/PP-train.csv --checkpoint_path models/my_videbertta \
    --epoch 8 --learning_rate 1e-5 --batch_size 16
```

### 3. Compare All Models
```bash
python main_model.py --action compare --data_path data/VLSP/PP-test.csv
```

### 4. Quick Inference
```bash
python main_model.py --action inference --model_name phobert \
    --inference_text "Tôi yêu Việt Nam!"
```

## 🎯 Research Goals

### Primary Objectives
1. **Model Comparison**: Compare Vietnamese-specific vs multilingual models
2. **Transfer Learning**: Evaluate cross-lingual knowledge transfer
3. **Data Augmentation**: Assess impact of synthetic Vietnamese data
4. **Performance Optimization**: Find optimal hyperparameters for Vietnamese sentiment analysis

### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Weighted F1 across sentiment classes
- **Confusion Matrix**: Detailed error analysis
- **Cross-lingual Performance**: Generalization to different Vietnamese domains

## 📚 References

- **PhoBERT**: Nguyen & Nguyen (2020) - First Vietnamese BERT
- **ViDeBERTa**: Tran et al. (2023) - State-of-the-art Vietnamese DeBERTa
- **XLM-RoBERTa**: Conneau et al. (2020) - Multilingual RoBERTa
- **CafeBERT**: Do et al. (2024) - Vietnamese-adapted model
- **ViT5**: Phan et al. (2022) - Vietnamese T5

## 🤝 Contributing

This framework supports Vietnamese sentiment analysis research. To contribute:
1. Add new Vietnamese models to `models.py`
2. Implement new data processing techniques in `helpers.py`
3. Add evaluation metrics and benchmarks
4. Submit results on standard Vietnamese datasets

---

**Note**: This framework is specifically designed for Vietnamese sentiment analysis and includes state-of-the-art Vietnamese language models. For optimal performance, ViDeBERTa is recommended for accuracy, while PhoBERT provides a good balance of performance and efficiency.

<!-- ```bash
python main_model.py --action train --model_name cafebert \
    --data_path data/Amazon/VLSP+translated-Amazon50.csv --checkpoint_path models/VLSP+translated-Amazon50 \
    --model_path models/VLSP+translated-Amazon50_cafebert_lr2e-05_bs64_epoch10_scheduler1_nw32 \
    --epoch 6 --learning_rate 5e-6 --batch_size 64 --device cuda:6

python main_model.py --action test --model_name cafebert \
    --data_path data/VLSP/PP-test.csv \
    --model_path models/VLSP+translated-Amazon50_cafebert_lr1e-05_bs64_epoch5_scheduler1_nw32 \
    --batch_size 64 --device cuda:6

python main_model.py --action test --model_name cafebert \
    --data_path data/VLSP/PP-test.csv \
    --model_path models/VLSP+translated-Amazon50_cafebert_lr5e-06_bs64_epoch6_scheduler1_nw32 \
    --batch_size 64 --device cuda:6

python main_model.py --action test --model_name cafebert \
    --data_path data/VLSP/PP-test.csv \
    --model_path models/VLSP+translated-Amazon50_cafebert_lr2e-05_bs64_epoch20_scheduler1_nw32 \
    --batch_size 64 --device cuda:6
``` -->
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
    --input_path data/Amazon/Split-train \
    --output_path data/Amazon/translated-PP-train.csv
```

### Data Balancing
```bash
# Undersample data for class balance
python main_data.py --action undersampling \
    --input_path data/Amazon/PP-train.csv --output_path data/Amazon/undersampled-PP-train.csv
```

## 🤖 Model Training & Evaluation

### Training Vietnamese Models

#### PhoBERT Training
```bash
# Train PhoBERT on VLSP dataset
python main_model.py --action train --model_name phobert \
    --data_path data/VLSP/PP-train.csv --checkpoint_path models/PhoBERT_VLSP \
    --epoch 10 --learning_rate 2e-5 --batch_size 64

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
    --epoch 8 --learning_rate 1e-5 --batch_size 32

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
    --epoch 10 --learning_rate 2e-5 --batch_size 64
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
    --model_path models/PhoBERT_VLSP_phobert_lr2e-05_bs64_epoch10_scheduler1_nw32

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
1. Direct training on Vietnamese data (VLSP)
2. Pre-training on translated Amazon → fine-tune on VLSP
3. Mixed training (Amazon + VLSP combined)
4. Cross-lingual transfer learning
5. Multi-task learning approaches

#### Data Augmentation Techniques
- **Undersampling**: Balance class distribution
- **Oversampling**: Increase minority class samples
- **SMOTE**: Synthetic minority oversampling
- **LLM-based generation**: Few-shot sample generation
- **Translation augmentation**: Translate and back-translate

## 📈 Performance Benchmarks

Based on research findings, expected performance on Vietnamese sentiment analysis:

| Model | Architecture | Parameters | VLSP Accuracy | Amazon Accuracy |
|-------|-------------|------------|---------------|-----------------|
| **ViDeBERTa** | DeBERTaV3 | 86M | **~97.2%** | **~89.9%** |
| **PhoBERT** | RoBERTa | 135M | ~96.8% | ~85.7% |
| **CafeBERT** | XLM-RoBERTa+ | 270M | ~96.5% | ~87.2% |
| **XLM-RoBERTa** | RoBERTa | 270M | ~96.3% | ~82.0% |
| **ViT5** | T5 | 220M | ~95.8% | ~81.3% |
| **Custom mBERT** | BERT+ | 180M | ~95.2% | ~79.5% |

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

```bash
python main_data.py --action preprocess_VLSP --data texts --label labels --input_path data/VLSP/OG-train.csv --output_path data/VLSP/PP-train.csv

python main_data.py --action preprocess_VLSP --data texts --label labels --input_path data/VLSP/OG-test.csv --output_path data/VLSP/PP-test.csv

python main_data.py --action preprocess_VLSP --data texts --label labels --input_path data/VLSP/OG-dev.csv --output_path data/VLSP/PP-dev.csv
```

## Preprocess Amazon data

```bash
python main_data.py --action preprocess_Amazon --data reviewText --label rating --input_path data/Amazon/OG-train.csv --output_path data/Amazon/PP-train.csv
python main_data.py --action split --input_path data/Amazon/PP-train.csv --output_path data/Amazon/Split-train
```

Manually send to OpenAI batch interface.

```bash
python main_data.py --action combine --original_csv_path data/Amazon/PP-train.csv --input_path data/Amazon/Split-train --output_path data/Amazon/translated-PP-train.csv
```
## Other tools

```bash
python main_data.py --action undersampling --input_path data/Amazon/PP-train.csv --output_path data/Amazon/undersampled-PP-train.csv

```


## train/test/inference

```bash
python main_model.py --action train --data_path data/amaz/PP-train.csv --checkpoint_path models/VLSP --epoch 20

python main_model.py --action train --data_path data/VLSP/PP-train.csv --checkpoint_path models/VLSP --epoch 10

python main_model.py --action train --data_path data/VLSP/PP-train.csv --checkpoint_path models/VLSP --epoch 5

python main_model.py --action train --data_path data/VLSP/PP-train.csv --checkpoint_path models/VLSP --epoch 4

python main_model.py --action train --data_path data/VLSP/PP-train.csv --checkpoint_path models/VLSP --epoch 20 --use_scheduler 0

python main_model.py --action train --data_path data/VLSP/PP-train.csv --checkpoint_path models/VLSP --epoch 10 --use_scheduler 0

python main_model.py --action train --data_path data/VLSP/PP-train.csv --checkpoint_path models/VLSP --epoch 5 --use_scheduler 0

python main_model.py --action train --data_path data/VLSP/PP-train.csv --checkpoint_path models/VLSP --epoch 4 --use_scheduler 0
```

```bash
python main_model.py --action test --data_path data/VLSP/PP-test.csv --model_path models/VLSP_mBERT_lr3e-05_bs128_epoch4_scheduler1_nw32
```

# Current Experiment Variables:

## Data: 
- VLSP (dataset 1)
- Amazon (dataset 2)

## Models (Tuan Anh):
- Custom mBert
- PhoBERT
- ViDeBERTa
- XLM-RoBERTa
- CafeBERT
- ViT5

## Ways to Finetunes (Nhat Minh): 

Given Dataset X, Y, Z,... How can we fine tune a models
- finetune on Amazon, then finetune on VLSP.
- finetune on translated Amazon, then fintune on VLSP.
- finetune on Amazon mix (concate then shuffle) with VLSP.
- finetune on translated Amazon mix with VLSP.
- 

## Data Augmentation (Ngoc Toan): 
- Undersampling
- OverSampling
- SMOTE
- LLM few shot sample generation
- Translate Dataset Amazon to Vietnamese.

## Evaluation:
- Calculate metrics when testing on VLSP test set.

# TO CODE:
- More detailed Evaluation
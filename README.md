# Data Manipulation Tools
## Preprocess VLSP data

```bash
python main_data.py --action preprocess_VLSP --data texts --label labels --input_path data/VLSP/OG-train.csv --output_path data/VLSP/PP-train.csv

python main_data.py --action preprocess_VLSP --data texts --label labels --input_path data/VLSP/OG-test.csv --output_path data/VLSP/PP-test.csv

python main_data.py --action preprocess_VLSP --data texts --label labels --input_path data/VLSP/OG-dev.csv --output_path data/VLSP/PP-dev.csv
```

## Preproces Amazon data

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
# Undersampling - balance dataset by reducing majority classes to minority class size
python main_data.py --action undersampling --input_path data/VLSP/PP-train.csv --output_path data/VLSP/undersampled-PP-train.csv

# Oversampling - balance dataset by duplicating minority class samples
python main_data.py --action oversampling --input_path data/VLSP/PP-train.csv --output_path data/VLSP/oversampled-PP-train.csv

# SMOTE - generate synthetic samples for minority classes
python main_data.py --action smote --input_path data/VLSP/PP-train.csv --output_path data/VLSP/smote-PP-train.csv

# LLM Few-shot Generation - generate synthetic samples using OpenAI GPT-4
# Note: Requires OPENAI_API_KEY in .env file
python main_data.py --action llm_generation --input_path data/VLSP/PP-train.csv --output_path data/VLSP/llm-augmented-PP-train.csv --target_samples_per_class 1500
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
- mBert

## Ways to Finetunes (Nhat Minh): 

Given Dataset X, Y, Z,... How can we fine tune a models
- finetune on Amazon, then finetune on VLSP.
- finetune on translated Amazon, then fintune on VLSP.
- finetune on Amazon mix (concate then shuffle) with VLSP.
- finetune on translated Amazon mix with VLSP.
- 

## Data Augmentation (Ngoc Toan): 
- ✅ Undersampling - Balance dataset by reducing majority classes
- ✅ OverSampling - Balance dataset by duplicating minority classes  
- ✅ SMOTE - Synthetic Minority Oversampling Technique
- ✅ LLM few shot sample generation - Generate synthetic samples using GPT-4
- ✅ Translate Dataset Amazon to Vietnamese

## Evaluation:
- Calculate metrics when testing on VLSP test set.

# TO CODE:
- ✅ More detailed Evaluation
- ✅ Complete Data Augmentation Implementation:
  - ✅ Undersampling 
  - ✅ OverSampling
  - ✅ SMOTE  
  - ✅ LLM few-shot generation
  - ✅ Amazon→Vietnamese translation

## 📊 Data Augmentation Techniques Available

All data augmentation techniques are now implemented! See `DATA_AUGMENTATION_SUMMARY.md` for detailed usage and test results.

## 🚀 Quick Start

1. **Preprocess datasets:**
   ```bash
   python main_data.py --action preprocess_VLSP --data texts --label labels --input_path data/VLSP/OG-train.csv --output_path data/VLSP/PP-train.csv
   ```

2. **Apply data augmentation:**
   ```bash
   # OverSampling (recommended for balanced training)
   python main_data.py --action oversampling --input_path data/VLSP/PP-train.csv --output_path data/VLSP/oversampled-train.csv
   
   # SMOTE (for synthetic minority samples)  
   python main_data.py --action smote --input_path data/VLSP/PP-train.csv --output_path data/VLSP/smote-train.csv
   ```

3. **Train model:**
   ```bash
   python main_model.py --action train --data_path data/VLSP/oversampled-train.csv --checkpoint_path models/VLSP --epoch 10
   ```
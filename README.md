## Preprocess data

```bash
python main_data.py --action preprocess_VLSP --data texts --label labels --input_path data/VLSP/OG-train.csv --output_path data/VLSP/PP-train.csv

python main_data.py --action preprocess_VLSP --data texts --label labels --input_path data/VLSP/OG-test.csv --output_path data/VLSP/PP-test.csv

python main_data.py --action preprocess_VLSP --data texts --label labels --input_path data/VLSP/OG-dev.csv --output_path data/VLSP/PP-dev.csv
```

## train/ test/ inference/

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
- Undersampling
- OverSampling
- SMOTE
- LLM few shot sample generation
- Translate Dataset Amazon to Vietnamese.

## Evaluation:
- Calculate metrics when testing on VLSP test set.
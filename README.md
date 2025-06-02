## Preprocess data

```bash
python main_data.py --action preprocess_VLSP --data texts --label labels --input_path data/VLSP/OG-train.csv --output_path data/VLSP/PP-train.csv

python main_data.py --action preprocess_VLSP --data texts --label labels --input_path data/VLSP/OG-test.csv --output_path data/VLSP/PP-test.csv

python main_data.py --action preprocess_VLSP --data texts --label labels --input_path data/VLSP/OG-dev.csv --output_path data/VLSP/PP-dev.csv
```

```bash
python main_model.py --action train --data_path data/VLSP/PP-train.csv --checkpoint_path models/VLSP --epoch 10

python main_model.py --action train --data_path data/VLSP/PP-train.csv --checkpoint_path models/VLSP --epoch 10 --use_scheduler 0

python main_model.py --action train --data_path data/VLSP/PP-train.csv --checkpoint_path models/VLSP --epoch 5 --use_scheduler 0

python main_model.py --action train --data_path data/VLSP/PP-train.csv --checkpoint_path models/VLSP --epoch 4 --use_scheduler 0
```

```bash
python main_model.py --action test --data_path data/VLSP/PP-test.csv --model_path models/VLSP
python main_model.py --action test --data_path data/VLSP/PP-test.csv --model_path models/VLSP_lr3e-05_bs128_epoch10_schedulerTrue_nw32
python main_model.py --action test --data_path data/VLSP/PP-test.csv --model_path models/VLSP_lr3e-05_bs128_epoch5_schedulerTrue_nw32
python main_model.py --action test --data_path data/VLSP/PP-test.csv --model_path models/VLSP_lr3e-05_bs128_epoch4_schedulerTrue_nw32
python main_model.py --action test --data_path data/VLSP/PP-test.csv --model_path models/VLSP_lr3e-05_bs256_epoch4_schedulerTrue_nw32
```
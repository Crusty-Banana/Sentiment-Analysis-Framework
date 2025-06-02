## Preprocess data

```bash
python main_data.py --action preprocess_VLSP --data texts --label labels --input_path data/VLSP/OG-train.csv --output_path data/VLSP/PP-train.csv

python main_data.py --action preprocess_VLSP --data texts --label labels --input_path data/VLSP/OG-test.csv --output_path data/VLSP/PP-test.csv

python main_data.py --action preprocess_VLSP --data texts --label labels --input_path data/VLSP/OG-dev.csv --output_path data/VLSP/PP-dev.csv

```
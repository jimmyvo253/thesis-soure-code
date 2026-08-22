# Thesis Source Code: Reinforcement Learning for Spaced Repetition

## 1. Setup Environment
Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```

## 2. Preprocess Data
Run the preprocessing script to clean the raw data and extract transitions:
```bash
python preprocess_anki_10k.py
```
*(Note: Please ensure the raw data is placed in the `data/` directory before running. Due to GitHub size limits, datasets are provided separately via Google Drive.)*

## 3. Train Models
Run the training script to train the DQN agents:
```bash
python src/train.py
```

## 4. Evaluate & Plot Results
Run the evaluation script to simulate the learning process and benchmark against baselines:
```bash
python src/evaluate.py
```

## 5. Other Utilities
- `python run_paired_t_test.py`: Computes statistical significance between models.

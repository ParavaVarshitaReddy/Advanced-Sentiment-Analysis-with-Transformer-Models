# Sentiment Analysis with XLM-RoBERTa

## Overview
This project implements a sentiment analysis model using **XLM-RoBERTa** for text classification. It fine-tunes transformer-based models on a dataset of Amazon reviews to predict sentiment with high accuracy.

## Features
- Uses **XLM-RoBERTa** and **Microsoft DeBERTa Large** for sentiment classification.
- Implements **data preprocessing, tokenization, and fine-tuning**.
- Achieves **95% accuracy** with XLM-RoBERTa.
- Includes **early stopping and hyperparameter tuning** for optimization.
- Utilizes **PyTorch, Hugging Face Transformers, and the Trainer API**.

## Setup
### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Sentiment-Analysis-XLM-RoBERTa.git
cd Sentiment-Analysis-XLM-RoBERTa
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Training Script
```bash
python main.py
```

## Project Structure
```
├── README.md
├── requirements.txt
├── train.csv  (Dataset file - Add .gitignore if needed)
├── test.csv   (Dataset file - Add .gitignore if needed)
├── main.py  (Main script for training and evaluation)
├── utils.py (Helper functions for preprocessing and metrics)
├── config.yaml (Configuration file for hyperparameters)
├── models/
│   ├── final_model/  (Folder for saving trained model)
│   ├── tokenizer/  (Folder for saving tokenizer)
└── .gitignore
```

## Results
- Achieved **95% accuracy** using **XLM-RoBERTa**.
- Model successfully classifies sentiment with optimized fine-tuning.

## Future Improvements
- Experiment with **larger datasets** for better generalization.
- Deploy model as an **API for real-time inference**.
- Implement **more transformer architectures** to compare performance.

## Author
Parava Varshita

## License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

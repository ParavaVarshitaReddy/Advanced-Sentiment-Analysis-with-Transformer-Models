# utils.py
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def clean_text(text):
    """Clean and validate text data."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    return str(text).strip()

def load_data(train_path, test_path, train_size=10000, test_size=10000):
    """Load and prepare datasets with text cleaning."""
    train_df = pd.read_csv(train_path, header=None, names=['polarity', 'title', 'text'])
    test_df = pd.read_csv(test_path, header=None, names=['polarity', 'title', 'text'])
    
    for df in [train_df, test_df]:
        df['title'] = df['title'].apply(clean_text)
        df['text'] = df['text'].apply(clean_text)
        df.dropna(subset=['title', 'text'], inplace=True)
        df['combined_text'] = df.apply(lambda x: f"{x['title']} </s> {x['text']}" if x['title'] and x['text'] else x['title'] if x['title'] else x['text'], axis=1)
        df['labels'] = df['polarity'] - 1
    
    train_df = train_df.sample(n=min(train_size, len(train_df)), random_state=42)
    test_df = test_df.sample(n=min(test_size, len(test_df)), random_state=42)
    
    return DatasetDict({
        'train': Dataset.from_pandas(train_df[['combined_text', 'labels']]),
        'test': Dataset.from_pandas(test_df[['combined_text', 'labels']])
    })

def preprocess_function(examples, tokenizer, max_length=256):
    """Tokenize text examples."""
    texts = [str(text) for text in examples['combined_text']]
    tokenized = tokenizer(texts, truncation=True, max_length=max_length, padding='max_length')
    tokenized['labels'] = examples['labels']
    return tokenized

def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, and F1 score."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

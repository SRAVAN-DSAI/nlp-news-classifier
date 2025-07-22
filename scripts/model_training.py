
import os
import pandas as pd
import pickle
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import logging
import sys

# Add parent directory (nlp_news_pipeline) to sys.path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.spark_utils import create_spark_session

# Set Hugging Face logging verbosity
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_info()
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(p):
    """
    Computes accuracy for model evaluation.
    """
    predictions = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, predictions)}

def train_news_classifier(
    cleaned_parquet_path="data/cleaned_news_parquet",
    model_output_dir="models/trained_news_classifier",
    label_map_output_path="models/label_map.pkl",
    model_name="distilbert-base-uncased",
    num_train_epochs=3,
    train_batch_size=16,
    eval_batch_size=64,
    val_split_size=0.1,
    random_seed=42
):
    """
    Orchestrates the training of the news classification model.
    """
    print("\n--- Starting Model Training Pipeline ---")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    full_cleaned_parquet_path = os.path.join(project_root, cleaned_parquet_path)
    full_model_output_dir = os.path.join(project_root, model_output_dir)
    full_label_map_output_path = os.path.join(project_root, label_map_output_path)

    # 1. Load data from Parquet using Spark
    spark = create_spark_session(app_name="ModelTrainingDataLoader", executor_memory="3g", driver_memory="3g")
    if not os.path.exists(full_cleaned_parquet_path):
        print(f"Error: Cleaned Parquet data not found at '{full_cleaned_parquet_path}'. Please run data_processing.py.")
        spark.stop()
        return

    cleaned_data_df_spark = spark.read.parquet(full_cleaned_parquet_path)
    print(f"Total records loaded from Parquet: {cleaned_data_df_spark.count()}")

    # Convert to Pandas for Hugging Face (assuming it fits in memory)
    model_data_pd = cleaned_data_df_spark.select("cleaned_text", "category").toPandas()
    spark.stop()
    print("SparkSession stopped after data loading.")

    # 2. Prepare data for Hugging Face Trainer
    unique_categories = model_data_pd['category'].unique()
    label_map = {category: i for i, category in enumerate(unique_categories)}
    model_data_pd['labels'] = model_data_pd['category'].map(label_map)

    # Save label map
    os.makedirs(os.path.dirname(full_label_map_output_path), exist_ok=True)
    with open(full_label_map_output_path, 'wb') as f:
        pickle.dump(label_map, f)
    print(f"Label map saved to {full_label_map_output_path}: {label_map}")

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        list(model_data_pd['cleaned_text']),
        list(model_data_pd['labels']),
        test_size=val_split_size,
        random_state=random_seed, # MUST be the same as in Code Block 5
        stratify=list(model_data_pd['labels'])
    )
    print(f"Training set size: {len(train_texts)}, Validation set size: {len(val_texts)}")

    # 3. Load Tokenizer and Model
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=len(label_map))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device for training: {device}")

    # Tokenize datasets
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
    train_dataset = NewsDataset(train_encodings, train_labels)
    val_dataset = NewsDataset(val_encodings, val_labels)

    # 4. Configure and Run Trainer
    training_args = TrainingArguments(
        output_dir='./results', # Relative to current script execution, will be in scripts/results
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs', # Relative to current script execution, will be in scripts/logs
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="tensorboard"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    print("\nTraining model...")
    trainer.train()
    print("Model training complete.")

    eval_results = trainer.evaluate()
    print(f"Evaluation Results after training: {eval_results}")

    # 5. Save the trained model and tokenizer
    os.makedirs(full_model_output_dir, exist_ok=True)
    trainer.save_model(full_model_output_dir)
    tokenizer.save_pretrained(full_model_output_dir)
    print(f"Model and tokenizer saved to {full_model_output_dir}")

if __name__ == "__main__":
    train_news_classifier()
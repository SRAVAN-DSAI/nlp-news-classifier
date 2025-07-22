import os
import pandas as pd
import pickle
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm # For progress bars

# Local utility import
import sys
# Add parent directory (nlp_news_pipeline) to sys.path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.spark_utils import create_spark_session

# Hugging Face imports
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, Dataset


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

def evaluate_model_from_saved_files(
    cleaned_parquet_path="data/cleaned_news_parquet",
    model_dir="models/trained_news_classifier",
    label_map_file="models/label_map.pkl",
    predictions_output_path="data/validation_predictions_reloaded.csv",
    val_split_size=0.1,
    random_seed=42
):
    """
    Evaluates a trained model using saved files, without retraining.
    """
    print("\n--- Starting Model Evaluation from Saved Files ---")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    full_cleaned_parquet_path = os.path.join(project_root, cleaned_parquet_path)
    full_model_dir = os.path.join(project_root, model_dir)
    full_label_map_file = os.path.join(project_root, label_map_file)
    full_predictions_output_path = os.path.join(project_root, predictions_output_path)


    # --- Step 1: Re-create SparkSession to load cleaned data ---
    spark = create_spark_session(app_name="DataEvaluationLoader")
    if not os.path.exists(full_cleaned_parquet_path):
        print(f"Error: Cleaned Parquet data directory not found at '{full_cleaned_parquet_path}'.")
        print("Please ensure `data_processing.py` was run successfully to generate this data.")
        spark.stop()
        return

    cleaned_data_df_spark = spark.read.parquet(full_cleaned_parquet_path)
    print(f"Total records loaded from Parquet: {cleaned_data_df_spark.count()}")
    model_data_pd = cleaned_data_df_spark.select("cleaned_text", "category").toPandas()
    spark.stop()
    print("SparkSession stopped after data loading.")

    # --- Step 2: Load the saved label_map ---
    try:
        with open(full_label_map_file, 'rb') as f:
            loaded_label_map = pickle.load(f)
        print(f"Successfully loaded label map from {full_label_map_file}")
    except FileNotFoundError:
        print(f"Error: Label map file not found at {full_label_map_file}.")
        print("Please ensure `model_training.py` was run successfully to save the label map.")
        return

    id_to_label = {v: k for k, v in loaded_label_map.items()}
    print(f"Loaded Label Map: {loaded_label_map}")
    print(f"ID to Label Mapping: {id_to_label}")

    # --- Step 3: Re-perform train_test_split to get the exact validation set ---
    model_data_pd['labels'] = model_data_pd['category'].map(loaded_label_map)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        list(model_data_pd['cleaned_text']),
        list(model_data_pd['labels']),
        test_size=val_split_size,
        random_state=random_seed, # MUST be the same as in training
        stratify=list(model_data_pd['labels'])
    )
    print(f"Re-created validation set size: {len(val_texts)}")

    # --- Step 4: Load the saved model and tokenizer ---
    if not os.path.exists(full_model_dir) or not os.listdir(full_model_dir):
        print(f"Error: Trained model directory not found or empty at '{full_model_dir}'.")
        print("Please ensure `model_training.py` was run successfully to save the model.")
        return

    print(f"Loading trained model and tokenizer from {full_model_dir}...")
    device_for_inference = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for inference: {device_for_inference}")

    loaded_tokenizer = AutoTokenizer.from_pretrained(full_model_dir)
    loaded_model = AutoModelForSequenceClassification.from_pretrained(full_model_dir)
    loaded_model.to(device_for_inference)
    loaded_model.eval()

    # --- Step 5: Perform predictions on the validation set ---
    print("\n--- Performing predictions on the validation set ---")
    val_encodings_loaded = loaded_tokenizer(val_texts, truncation=True, padding=True, max_length=128)
    val_dataset_loaded = NewsDataset(val_encodings_loaded, val_labels)
    val_loader = DataLoader(val_dataset_loaded, batch_size=64, shuffle=False)

    all_preds = []
    all_labels_from_loader = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Predicting on validation set"):
            input_ids = batch['input_ids'].to(loaded_model.device)
            attention_mask = batch['attention_mask'].to(loaded_model.device)
            labels = batch['labels'].to(loaded_model.device)

            outputs = loaded_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels_from_loader.extend(labels.cpu().numpy())

    # --- Step 6: Calculate and display detailed metrics ---
    print("\nDetailed Classification Report on Validation Set:")
    target_names = [id_to_label[i] for i in sorted(id_to_label.keys())]
    print(classification_report(all_labels_from_loader, all_preds, target_names=target_names))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels_from_loader, all_preds)
    print(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # --- Step 7: Error Analysis (Sample Misclassified Examples) ---
    print("\n--- Sample Misclassified Examples ---")
    misclassified_indices = np.where(np.array(all_labels_from_loader) != np.array(all_preds))[0]

    if len(misclassified_indices) > 0:
        print(f"Total misclassified examples: {len(misclassified_indices)}")
        sample_errors_to_show = 5
        for i, idx in enumerate(misclassified_indices[:min(sample_errors_to_show, len(misclassified_indices))]):
            true_label_name = id_to_label.get(all_labels_from_loader[idx], "Unknown")
            predicted_label_name = id_to_label.get(all_preds[idx], "Unknown")
            print(f"\n--- Misclassified Example {i+1} ---")
            print(f"Text: {val_texts[idx][:200]}...")
            print(f"True Label: {true_label_name}, Predicted Label: {predicted_label_name}")
    else:
        print("No misclassified examples found on the validation set.")

    # --- Step 8: Exporting Predictions for External Analysis ---
    print("\n--- Exporting Predictions for External Analysis ---")
    predictions_df = pd.DataFrame({
        'text': val_texts,
        'true_label_idx': all_labels_from_loader,
        'predicted_label_idx': all_preds,
        'true_category': [id_to_label.get(label, "Unknown") for label in all_labels_from_loader],
        'predicted_category': [id_to_label.get(pred, "Unknown") for pred in all_preds]
    })

    # Save to a path relative to the project root
    os.makedirs(os.path.dirname(full_predictions_output_path), exist_ok=True)
    predictions_df.to_csv(full_predictions_output_path, index=False)
    print(f"Validation predictions saved to: {full_predictions_output_path}")

    # --- Step 9: Make predictions on new example texts (manual input) ---
    print("\n--- Predictions on new example texts (manual input) ---")
    example_texts_manual = [
        "Manchester United wins Premier League title in dramatic fashion.",
        "Stock market crashes amid fears of recession and inflation.",
        "NASA launches new telescope to explore distant galaxies.",
        "Diplomats meet to discuss new international trade agreements."
    ]

    for text in example_texts_manual:
        inputs = loaded_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device_for_inference)
        with torch.no_grad():
            outputs = loaded_model(**inputs)
            logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_label_idx = torch.argmax(probabilities, dim=-1).item()
        predicted_score = probabilities[0, predicted_label_idx].item()
        predicted_category = id_to_label.get(predicted_label_idx, "Unknown Category")
        print(f"Text: \"{text}\"")
        print(f"Predicted Category: {predicted_category} (Confidence: {predicted_score:.4f})")

    print("\n--- End of Model Analysis from Saved Files ---")

if __name__ == "__main__":
    evaluate_model_from_saved_files()
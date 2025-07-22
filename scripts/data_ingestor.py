import json
import os
from datetime import datetime
from datasets import load_dataset

def ingest_real_news_data(dataset_name="ag_news", split="train", num_examples=None, output_dir="data/raw_news"):
    """
    Ingests a real news dataset from Hugging Face Hub and saves it locally.
    """
    print(f"Attempting to load real dataset: '{dataset_name}' (split: '{split}')...")
    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"Error loading dataset '{dataset_name}': {e}")
        print("Ensure the dataset name is correct and you have an internet connection.")
        return None

    if num_examples:
        dataset = dataset.select(range(min(num_examples, len(dataset))))
        print(f"Processing a subset of {len(dataset)} examples.")
    else:
        print(f"Processing all {len(dataset)} examples in the '{split}' split.")

    # Create data directory if it doesn't exist (relative to project root)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    full_output_dir = os.path.join(project_root, output_dir)
    os.makedirs(full_output_dir, exist_ok=True)

    output_filepath = os.path.join(full_output_dir, f"{dataset_name}_{split}.jsonl")

    processed_count = 0
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for i, record in enumerate(dataset):
            category_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

            current_timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            processed_record = {
                "id": i,
                "timestamp": current_timestamp_str,
                "text": record.get("text", ""),
                "source": dataset_name,
                "category": category_map.get(record.get("label"), "unknown")
            }
            f.write(json.dumps(processed_record) + '\n')
            processed_count += 1

            if (processed_count % (len(dataset) // 10) == 0) and (len(dataset) // 10 > 0):
                print(f"Saved {processed_count}/{len(dataset)} records...")
            elif processed_count == len(dataset):
                print(f"Saved all {processed_count}/{len(dataset)} records.")

    print(f"Real-world news data ingested and saved to: {output_filepath}")
    return output_filepath

if __name__ == "__main__":
    print("\n--- Running Data Ingestion Script ---")
    ingested_file = ingest_real_news_data(num_examples=120000, output_dir="data/raw_news")
    if ingested_file:
        print(f"Data ingestion completed. Data available at: {ingested_file}")
    else:
        print("Data ingestion failed.")
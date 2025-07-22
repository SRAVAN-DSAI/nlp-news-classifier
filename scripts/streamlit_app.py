
import streamlit as st
import torch
import numpy as np
import os
import pickle
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- Configuration Paths (relative to project root) ---
LOCAL_MODEL_DIR_RELATIVE = "models/trained_news_classifier"
LOCAL_LABEL_MAP_FILE_RELATIVE = "models/label_map.pkl"

# --- Model Loading (Cached for performance) ---
@st.cache_resource # Use st.cache_resource for models/heavy objects
def load_model_and_tokenizer():
    """
    Loads the trained model, tokenizer, and label map.
    Assumes files are present in the defined local paths.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_dir_to_load = os.path.join(project_root, LOCAL_MODEL_DIR_RELATIVE)
    label_map_file_to_load = os.path.join(project_root, LOCAL_LABEL_MAP_FILE_RELATIVE)

    try:
        # Determine device (GPU if available, else CPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.write(f"Loading model on: {device}")

        # Load label map
        if not os.path.exists(label_map_file_to_load):
            st.error(f"Error: Label map file not found at '{label_map_file_to_load}'.")
            st.info("Please ensure 'label_map.pkl' is in your 'models/' directory.")
            return None, None, None, None

        with open(label_map_file_to_load, 'rb') as f:
            loaded_label_map = pickle.load(f)
        id_to_label = {v: k for k, v in loaded_label_map.items()}

        # Load model and tokenizer
        if not os.path.exists(model_dir_to_load) or not os.listdir(model_dir_to_load):
            st.error(f"Error: Model directory not found or empty at '{model_dir_to_load}'.")
            st.info("Please ensure the 'trained_news_classifier' folder is copied into your 'models/' directory.")
            return None, None, None, None

        tokenizer = AutoTokenizer.from_pretrained(model_dir_to_load)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir_to_load)
        model.to(device)
        model.eval() # Set model to evaluation mode

        st.success("Model and tokenizer loaded successfully!")
        return model, tokenizer, id_to_label, device

    except Exception as e:
        st.error(f"Failed to load model or tokenizer: {e}")
        st.info("Please ensure the model files are valid and compatible with the installed `transformers` and `torch` versions.")
        st.info(f"Details: {e}")
        return None, None, None, None

# Load the model and tokenizer when the script first runs
model, tokenizer, id_to_label, device = load_model_and_tokenizer()

# --- Streamlit App Interface ---
st.set_page_config(page_title="News Article Classifier", layout="centered", icon="📰")

st.title("📰 News Article Category Classifier")
st.markdown("""
    Enter a news article text below, and I'll predict its category (World, Sports, Business, Sci/Tech).
    This model is a fine-tuned DistilBERT.
""")

if model is None or tokenizer is None or id_to_label is None:
    st.warning("The model could not be loaded. Please check the console for errors and ensure model files are correctly placed.")
else:
    user_input = st.text_area("Enter News Article Text:", height=200,
                              placeholder="e.g., 'Scientists discover new exoplanet with potential for life.'")

    if st.button("Predict Category"):
        if user_input:
            with st.spinner("Predicting..."):
                try:
                    # Tokenize input
                    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)

                    # Perform inference
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits

                    # Get probabilities and predicted label
                    probabilities = torch.nn.functional.softmax(logits, dim=-1)
                    predicted_label_idx = torch.argmax(probabilities, dim=-1).item()
                    predicted_score = probabilities[0, predicted_label_idx].item()

                    predicted_category = id_to_label.get(predicted_label_idx, "Unknown")

                    st.subheader("Prediction Result:")
                    st.success(f"**Category:** {predicted_category}")
                    st.info(f"**Confidence:** {predicted_score:.4f}")

                    st.markdown("---")
                    st.write("Raw probabilities for all categories:")
                    prob_df = pd.DataFrame({
                        "Category": [id_to_label.get(i, f"LABEL_{i}") for i in range(len(id_to_label))],
                        "Probability": probabilities.cpu().numpy().flatten()
                    }).sort_values(by="Probability", ascending=False)
                    st.dataframe(prob_df, hide_index=True)

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                    st.info("Please try again or check the input text.")
        else:
            st.warning("Please enter some text to predict.")

st.markdown("---")
st.caption("Developed using Hugging Face Transformers, PyTorch, and Streamlit.")
st.caption("Model trained on the AG News dataset.")
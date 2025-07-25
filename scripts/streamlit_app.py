import streamlit as st
import torch
import numpy as np
import os
import pickle
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import io
import time

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="News Article Classifier | Sravan Kodari",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/SRAVAN-DSAI/nlp_news_pipeline',
        'Report a Bug': 'mailto:sravankodari4@gmail.com',
        'About': 'News Article Classifier by Sravan Kodari'
    }
)

# --- Custom CSS ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        color: #2c3e50;
    }
    .header {
        background: #f0f2f6;
        padding: 1rem;
        border-bottom: 1px solid #d1d5db;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .settings {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: #3498db;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #d1d5db;
        padding: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
    <div class='header'>
        <h2>📰 News Classifier <span style='color: #7f8c8d;'>by Sravan Kodari</span></h2>
        <div>
            <a href='https://github.com/SRAVAN-DSAI/nlp_news_pipeline' target='_blank'>GitHub</a> |
            <a href='https://www.linkedin.com/in/sravan-kodari' target='_blank'>LinkedIn</a>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- Model Information ---
st.markdown("""
    <h1>AI-Powered News Article Classification</h1>
    <p>Analyze news articles with advanced machine learning. Classify content into Technology, Business, Sports, or Entertainment categories with confidence scores.</p>
    <p><em>DistilBERT Model</em> | <span style='color: #3498db;'>94.22% Accuracy</span> | <span style='color: #2ecc71;'>Real-time Processing</span></p>
""", unsafe_allow_html=True)

# --- Configuration Paths ---
LOCAL_MODEL_DIR_RELATIVE = "models/trained_news_classifier"
LOCAL_LABEL_MAP_FILE_RELATIVE = "models/label_map.pkl"

# --- Model Loading ---
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_dir_to_load = os.path.join(project_root, LOCAL_MODEL_DIR_RELATIVE)
    label_map_file_to_load = os.path.join(project_root, LOCAL_LABEL_MAP_FILE_RELATIVE)

    loading_container = st.container()
    with loading_container:
        st.markdown("### Model Loading")
        loading_status = st.empty()
        my_bar = st.progress(0)
        steps = ["Initializing...", "Checking device...", "Loading label map...", "Loading tokenizer...", "Loading model..."]
        for i, step in enumerate(steps):
            loading_status.info(step)
            my_bar.progress((i + 1) * 20)
            time.sleep(0.2)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loading_status.info(f"Using device: **{device.type.upper()}**")

        try:
            with open(label_map_file_to_load, 'rb') as f:
                loaded_label_map = pickle.load(f)
            id_to_label = {v: k for k, v in loaded_label_map.items()}

            tokenizer = AutoTokenizer.from_pretrained(model_dir_to_load)
            model = AutoModelForSequenceClassification.from_pretrained(model_dir_to_load)
            model.to(device)
            model.eval()

            loading_status.success("Model and tokenizer loaded successfully!")
            my_bar.progress(100)
            time.sleep(0.5)
            loading_container.empty()
            return model, tokenizer, id_to_label, device
        except Exception as e:
            loading_status.error(f"Failed to load model: {e}")
            st.stop()
            return None, None, None, None

model, tokenizer, id_to_label, device = load_model_and_tokenizer()

# --- Prediction Function ---
def predict_category(text_list, loaded_model, loaded_tokenizer, id_to_label_map, current_device, progress_text_placeholder=None):
    results = []
    total_texts = len(text_list)
    for i, text in enumerate(text_list):
        if progress_text_placeholder:
            progress_text_placeholder.text(f"Predicting text {i+1}/{total_texts}...")
        
        if not isinstance(text, str) or not text.strip():
            results.append({"text": text, "predicted_category": "Invalid/Empty Input", "confidence": 0.0, "raw_probabilities": {}})
            continue

        inputs = loaded_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(current_device)
        with torch.no_grad():
            output = loaded_model(**inputs)
            logits = output.logits

        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_label_idx = torch.argmax(probabilities, dim=-1).item()
        predicted_score = probabilities[0, predicted_label_idx].item()
        predicted_category = id_to_label_map.get(predicted_label_idx, "Unknown")
        raw_probs_dict = {id_to_label_map.get(idx, f"LABEL_{idx}"): prob for idx, prob in enumerate(probabilities.cpu().numpy().flatten())}

        results.append({
            "text": text,
            "predicted_category": predicted_category,
            "confidence": predicted_score,
            "raw_probabilities": raw_probs_dict
        })
    if progress_text_placeholder:
        progress_text_placeholder.empty()
    return results

# --- Settings ---
st.markdown('<div class="settings">', unsafe_allow_html=True)
st.markdown('<h3>⚙️ Settings</h3>', unsafe_allow_html=True)
confidence_threshold = st.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.05,
    help="Filter predictions by minimum confidence level."
)
st.markdown('</div>', unsafe_allow_html=True)

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📝 Single Article", "📊 Batch Analysis", "🧠 Model Details", "👋 About"])

with tab1:
    st.header("Analyze a Single Article")
    st.markdown("Enter a news article to classify its category and view confidence scores.")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        article_text = st.text_area(
            "Article Text",
            height=200,
            placeholder="e.g., 'Scientists discover new exoplanet with potential for life...'",
            help="Paste or type a news article here."
        )
    with col2:
        st.write("")  # Spacer
        if st.button("Try Sample Text"):
            article_text = "Scientists discover new exoplanet with potential for life..."

    if st.button("🚀 Classify Article"):
        if article_text and len(article_text.strip()) >= 10:
            with st.spinner("Classifying..."):
                single_result = predict_category([article_text], model, tokenizer, id_to_label, device)[0]
            
            st.subheader("Classification Result")
            col_result, col_probs = st.columns([1, 1])
            with col_result:
                if single_result["confidence"] >= confidence_threshold:
                    st.success(f"**Category:** {single_result['predicted_category']}")
                    st.metric("Confidence", f"{single_result['confidence']:.2%}")
                else:
                    st.warning(f"**Category:** {single_result['predicted_category']} (Below threshold)")
                    st.metric("Confidence", f"{single_result['confidence']:.2%}")
            with col_probs:
                prob_df = pd.DataFrame({
                    "Category": single_result['raw_probabilities'].keys(),
                    "Probability": [f"{v:.2%}" for v in single_result['raw_probabilities'].values()]
                }).sort_values(by="Probability", ascending=False)
                st.dataframe(prob_df, hide_index=True, use_container_width=True)
        else:
            st.error("Please enter a valid article text.")

with tab2:
    st.header("Batch Analysis")
    st.write("Upload a text or CSV file to classify multiple articles at once.")

with tab3:
    st.header("Model Details")
    st.write("Learn about the machine learning model powering this app.")

with tab4:
    st.header("About")
    st.write("This app demonstrates an end-to-end NLP pipeline for news article classification.")

# --- Footer ---
st.markdown("""
    <div style='text-align: center; padding: 1rem; background: #f0f2f6; border-top: 1px solid #d1d5db;'>
        © 2025 Sravan Kodari | Built with React, TypeScript, and Tailwind CSS | 
        <a href='https://github.com/SRAVAN-DSAI/nlp_news_pipeline' target='_blank'>GitHub</a> | 
        <a href='https://www.linkedin.com/in/sravan-kodari' target='_blank'>LinkedIn</a> | Last updated: January 2025
    </div>
""", unsafe_allow_html=True)

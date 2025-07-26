import streamlit as st
import torch
import numpy as np
import os
import pickle
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import io
import time
import plotly.graph_objects as go

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="News Article Classifier | Sravan Kodari",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/SRAVAN-DSAI/nlp_news_pipeline',
        'Report a Bug': 'mailto:sravankodari4@gmail.com',
        'About': 'News Article Classifier by Sravan Kodari'
    }
)

# --- Custom CSS for a Polished UI ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    :root {
        /* Light Mode Variables (Enforced) */
        --primary-bg: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        --secondary-bg: #f0f2f6;
        --text-color: #2c3e50;
        --accent-color: #3498db;
        --button-bg: linear-gradient(45deg, #3498db, #2980b9);
        --button-hover-bg: linear-gradient(45deg, #2980b9, #3498db);
        --button-text: #ffffff;
        --border-color: #d1d5db;
        --shadow-color: rgba(0,0,0,0.1);
        --alert-success-bg: #e6f4ea;
        --alert-success-text: #2e7d32;
        --alert-warning-bg: #fff3e0;
        --alert-warning-text: #ef6c00;
    }

    .stApp {
        font-family: 'Inter', sans-serif;
        background: var(--primary-bg);
        color: var(--text-color);
        transition: all 0.3s ease;
    }

    .stSidebar {
        background-color: var(--secondary-bg);
        border-right: 1px solid var(--border-color);
        box-shadow: 2px 0 5px var(--shadow-color);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px; padding: 10px; background-color: var(--secondary-bg);
        border-radius: 12px; box-shadow: 0 2px 4px var(--shadow-color);
    }
    .stTabs [data-baseweb="tab-list"] button {
        padding: 12px 20px; border-radius: 10px; background-color: transparent;
        transition: all 0.3s ease; font-weight: 600; color: var(--text-color);
    }
    .stTabs [data-baseweb="tab-list"] button:hover {
        background-color: var(--border-color); transform: translateY(-2px);
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: var(--accent-color); color: #ffffff;
        box-shadow: 0 2px 4px var(--shadow-color);
    }

    .stButton>button {
        background: var(--button-bg); color: var(--button-text); border: none;
        border-radius: 12px; padding: 12px 24px; font-size: 1.1rem;
        font-weight: 600; transition: all 0.3s ease; box-shadow: 0 2px 4px var(--shadow-color);
    }
    .stButton>button:hover {
        background: var(--button-hover-bg); transform: translateY(-2px);
        box-shadow: 0 4px 8px var(--shadow-color);
    }

    .stTextArea textarea {
        border-radius: 12px; border: 2px solid var(--border-color); padding: 12px;
        font-size: 1rem; transition: border-color 0.3s ease; background-color: var(--secondary-bg);
        color: var(--text-color);
    }
    .stTextArea textarea:focus {
        border-color: var(--accent-color); box-shadow: 0 0 5px rgba(52, 152, 219, 0.5);
    }

    .stDataFrame {
        border-radius: 12px; overflow: hidden; box-shadow: 0 2px 4px var(--shadow-color);
        background-color: var(--secondary-bg);
    }

    h1, h2, h3 {
        color: var(--text-color); font-weight: 700;
    }

    a {
        color: var(--accent-color); text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
    <div style='background-color:var(--secondary-bg); padding:1rem; border-radius:12px; margin-bottom:1rem;
                display:flex; justify-content:space-between; align-items:center; box-shadow: 0 2px 4px var(--shadow-color);'>
        <div>
            <h2 style='margin:0;'>📰 News Classifier</h2>
            <span style='color:var(--text-color); opacity:0.7;'>by Sravan Kodari</span>
        </div>
        <div>
            <a href='https://github.com/SRAVAN-DSAI/nlp_news_pipeline' target='_blank' style='margin-right:1.5rem;'>GitHub</a>
            <a href='https://www.linkedin.com/in/sravan-kodari' target='_blank'>LinkedIn</a>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- Configuration Paths for Streamlit Cloud ---
# These paths are relative to the root of your GitHub repository.
MODEL_DIR = "models/trained_news_classifier"
LABEL_MAP_FILE = "models/label_map.pkl"

# --- Model Loading ---
@st.cache_resource(show_spinner="Loading classification model...")
def load_model_and_tokenizer():
    """
    Loads the model, tokenizer, and label map.
    Uses st.cache_resource to load only once.
    Handles errors for deployment on Streamlit Cloud.
    """
    # Check if model files exist
    if not os.path.exists(MODEL_DIR) or not os.path.exists(LABEL_MAP_FILE):
        st.error(
            f"**Error:** Model files not found. 🚨**\n\n"
            f"Please ensure the `{MODEL_DIR}` and `{LABEL_MAP_FILE}` are in your GitHub repository.\n\n"
            f"**Important**: If your model is large, you must use Git LFS (Large File Storage) to track it. "
        )
        st.stop()

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open(LABEL_MAP_FILE, 'rb') as f:
            loaded_label_map = pickle.load(f)
        id_to_label = {v: k for k, v in loaded_label_map.items()}

        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        model.to(device)
        model.eval()

        return model, tokenizer, id_to_label, device
    except Exception as e:
        st.error(f"An error occurred during model loading: {e}")
        st.stop()

model, tokenizer, id_to_label, device = load_model_and_tokenizer()

# --- Prediction Function ---
def predict_category(text_list, loaded_model, loaded_tokenizer, id_to_label_map, current_device):
    """Predicts categories for a list of texts."""
    results = []
    if not text_list:
        return results

    for text in text_list:
        if not isinstance(text, str) or not text.strip():
            results.append({"text": text, "predicted_category": "Invalid/Empty", "confidence": 0.0, "raw_probabilities": {}})
            continue

        inputs = loaded_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(current_device)
        with torch.no_grad():
            logits = loaded_model(**inputs).logits

        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_label_idx = torch.argmax(probabilities, dim=-1).item()
        
        results.append({
            "text": text,
            "predicted_category": id_to_label_map.get(predicted_label_idx, "Unknown"),
            "confidence": probabilities[0, predicted_label_idx].item(),
            "raw_probabilities": {id_to_label_map.get(i, f"L_{i}"): p for i, p in enumerate(probabilities.cpu().numpy().flatten())}
        })
    return results

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold", min_value=0.0, max_value=1.0, value=0.75, step=0.05,
        help="Filter predictions by minimum confidence level.", key="confidence_threshold"
    )
    max_batch_size = st.number_input(
        "Max Batch Size", min_value=1, max_value=1000, value=100, step=10,
        help="Limit the number of articles processed in one batch from a file."
    )
    if st.button("🧹 Clear Inputs & Results", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# --- Main App ---
tab1, tab2, tab3 = st.tabs(["📝 **Single Article**", "📊 **Batch Analysis**", "🧠 **Model & About**"])

# --- Tab 1: Single Article Analysis ---
with tab1:
    st.header("Analyze a Single Article")
    sample_article = "Google announced today that Sundar Pichai will lead a new AI initiative in London. The company's stock rose slightly after the announcement."
    
    # Define the callback function to update the state
    def load_sample():
        st.session_state.text_input_single = sample_article

    col_input, col_sample = st.columns([3, 1])
    with col_input:
        # This widget's key is 'text_input_single'
        user_input_single = st.text_area(
            "Article Text", height=200, placeholder="e.g., 'Scientists discover new exoplanet with potential for life.'",
            help="Paste or type a news article here.", key="text_input_single"
        )
    with col_sample:
        st.write("")
        st.write("")
        # Use the on_click callback here. The button no longer needs an 'if' statement.
        st.button(
            "Try Sample Article",
            on_click=load_sample, # This runs before the script reruns
            use_container_width=True
        )

    if st.button("🚀 Classify Article", type="primary", use_container_width=True, disabled=not user_input_single):
        if len(user_input_single.strip()) < 10:
            st.warning("Input is too short. Please provide more text for an accurate classification.")
        else:
            with st.spinner("Classifying..."):
                result = predict_category([user_input_single], model, tokenizer, id_to_label, device)[0]
            
            st.subheader("Classification Result")
            col_result, col_probs = st.columns(2)
            
            with col_result:
                category = result['predicted_category']
                confidence = result['confidence']
                if confidence >= confidence_threshold:
                    st.success(f"**Category: {category}**")
                else:
                    st.warning(f"**Category: {category}** (Confidence below threshold)")
                st.metric("Confidence Score", f"{confidence:.2%}")

            with col_probs:
                # --- Plotly Visualization ---
                probs_df = pd.DataFrame(list(result['raw_probabilities'].items()), columns=['Category', 'Probability'])
                probs_df = probs_df.sort_values(by='Probability', ascending=True)

                fig = go.Figure(go.Bar(
                    x=probs_df['Probability'],
                    y=probs_df['Category'],
                    orientation='h',
                    marker_color='#3498db'
                ))
                fig.update_layout(
                    title="Probability Distribution",
                    xaxis_title="Probability",
                    yaxis_title="",
                    height=250,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='var(--text-color)'),
                    xaxis=dict(tickformat=".0%")
                )
                st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: Batch Analysis ---
with tab2:
    st.header("Analyze a Batch of Articles")
    st.markdown("Upload a **TXT** (one article per line) or **CSV** (must contain a 'text' column) file.")
    
    uploaded_file = st.file_uploader(
        "Upload your file", type=["txt", "csv"], key="file_uploader"
    )

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.txt'):
                texts = [line.decode("utf-8").strip() for line in uploaded_file if line.strip()]
            elif uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                if 'text' not in df.columns:
                    st.error("CSV file must contain a 'text' column.")
                    st.stop()
                texts = df['text'].dropna().astype(str).tolist()

            st.info(f"Found **{len(texts)}** articles in the file.")

            if len(texts) > max_batch_size:
                st.warning(f"File contains {len(texts)} articles. Only processing the first **{max_batch_size}** as per settings.")
                texts = texts[:max_batch_size]
            
            if st.button(f"📊 Run Analysis on {len(texts)} Articles", use_container_width=True, type="primary"):
                progress_bar = st.progress(0, text="Starting batch analysis...")
                results = []
                for i, text_chunk in enumerate(np.array_split(texts, (len(texts) // 10) + 1)):
                    results.extend(predict_category(text_chunk.tolist(), model, tokenizer, id_to_label, device))
                    progress_bar.progress((i + 1) / len(np.array_split(texts, (len(texts) // 10) + 1)), f"Processing... {len(results)}/{len(texts)} articles classified.")
                
                progress_bar.empty()
                st.success("Batch analysis complete!")
                
                results_df = pd.DataFrame(results)
                st.session_state.results_df = results_df
                
        except Exception as e:
            st.error(f"Error processing file: {e}")

    if 'results_df' in st.session_state:
        st.subheader("Analysis Results")
        results_df = st.session_state.results_df
        
        # --- Plotly Visualization for Batch ---
        filtered_df = results_df[results_df['confidence'] >= confidence_threshold]
        category_counts = filtered_df['predicted_category'].value_counts()
        
        if not category_counts.empty:
            col_chart, col_summary = st.columns(2)
            with col_chart:
                fig = go.Figure(go.Bar(
                    x=category_counts.index,
                    y=category_counts.values,
                    marker_color=['#3498db', '#e74c3c', '#2ecc71', '#f1c40f']
                ))
                fig.update_layout(
                    title=f"Category Distribution (Confidence ≥ {confidence_threshold:.0%})",
                    xaxis_title="Category",
                    yaxis_title="Article Count",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='var(--text-color)')
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_summary:
                st.dataframe(category_counts.reset_index().rename(columns={'index': 'Category', 'predicted_category': 'Count'}), use_container_width=True)
            
            # --- Download Button ---
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Results (CSV)",
                data=csv,
                file_name='news_classification_results.csv',
                mime='text/csv',
                use_container_width=True
            )
        else:
            st.warning("No articles met the confidence threshold. Try lowering it in the sidebar settings.")

# --- Tab 3: Model & About ---
with tab3:
    st.header("🧠 Model & App Details")
    st.markdown(f"""
    This app uses a **DistilBERT** model fine-tuned for multi-class text classification on the **AG News** dataset. 
    It classifies articles into one of four categories: **{', '.join(id_to_label.values())}**.
    - **Device In Use:** `{device.type.upper()}`
    - **Model Source:** Hugging Face Transformers
    - **Frontend:** Streamlit
    
    ### Performance Metrics (on test set)
    """)
    
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", "94.22%")
    col2.metric("F1-Score (Macro)", "94.00%")
    
    st.header("👋 About the Creator")
    st.markdown("""
    This app was built by **Sravan Kodari** to demonstrate an end-to-end NLP pipeline.
    - **Connect:** [GitHub](https://github.com/SRAVAN-DSAI/nlp_news_pipeline) | [LinkedIn](https://www.linkedin.com/in/sravan-kodari)
    - **Contact:** sravankodari4@gmail.com
    
    <small>_Last updated: July 2025_</small>
    """, unsafe_allow_html=True)

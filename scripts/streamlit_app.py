import streamlit as st
import torch
import numpy as np
import os
import pickle
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import io
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import uuid
from PIL import Image
import base64

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="News Article Classifier | Sravan Kodari",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/SRAVAN-DSAI/nlp_news_pipeline',
        'Report a Bug': 'mailto:sravankodari4@gmail.com',
        'About': 'Advanced News Article Classifier by Sravan Kodari'
    }
)

# --- Custom CSS for Enhanced Professional Look ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

    :root {
        --primary-bg: linear-gradient(135deg, #e6e9f0 0%, #eef1f5 100%);
        --secondary-bg: #ffffff;
        --text-color: #1a202c;
        --accent-color: #3b82f6;
        --button-bg: linear-gradient(45deg, #3b82f6, #2563eb);
        --button-hover-bg: linear-gradient(45deg, #2563eb, #3b82f6);
        --button-text: #ffffff;
        --border-color: #e2e8f0;
        --shadow-color: rgba(0,0,0,0.15);
        --alert-success-bg: #d1fae5;
        --alert-success-text: #065f46;
        --alert-warning-bg: #fef3c7;
        --alert-warning-text: #92400e;
        --chart-bg: #ffffff;
        --chart-text: #1a202c;
    }

    .stApp {
        font-family: 'Roboto', sans-serif;
        background: var(--primary-bg);
        color: var(--text-color);
        transition: all 0.3s ease;
    }

    .stSidebar {
        background-color: var(--secondary-bg);
        border-right: 2px solid var(--border-color);
        box-shadow: 3px 0 8px var(--shadow-color);
        padding: 1rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        padding: 12px;
        background-color: var(--secondary-bg);
        border-radius: 8px;
        box-shadow: 0 3px 6px var(--shadow-color);
    }

    .stTabs [data-baseweb="tab-list"] button {
        padding: 10px 18px;
        border-radius: 8px;
        background-color: transparent;
        transition: all 0.3s ease;
        font-weight: 500;
        color: var(--text-color);
        border: 1px solid var(--border-color);
    }

    .stTabs [data-baseweb="tab-list"] button:hover {
        background-color: var(--accent-color);
        color: #ffffff;
        transform: translateY(-1px);
    }

    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: var(--accent-color);
        color: #ffffff;
        box-shadow: 0 2px 4px var(--shadow-color);
        border: none;
    }

    .stButton>button {
        background: var(--button-bg);
        color: var(--button-text);
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px var(--shadow-color);
    }

    .stButton>button:hover {
        background: var(--button-hover-bg);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px var(--shadow-color);
    }

    .stTextArea textarea {
        border-radius: 8px;
        border: 2px solid var(--border-color);
        padding: 12px;
        font-size: 1rem;
        transition: border-color 0.3s ease;
        background-color: var(--secondary-bg);
        color: var(--text-color);
    }

    .stTextArea textarea:focus {
        border-color: var(--accent-color);
        box-shadow: 0 0 8px rgba(59, 130, 246, 0.3);
    }

    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 3px 6px var(--shadow-color);
        background-color: var(--secondary-bg);
    }

    .stAlert[data-testid="stAlertSuccess"] {
        background-color: var(--alert-success-bg);
        color: var(--alert-success-text);
        border-radius: 8px;
        padding: 16px;
        font-weight: 500;
    }

    .stAlert[data-testid="stAlertWarning"] {
        background-color: var(--alert-warning-bg);
        color: var(--alert-warning-text);
        border-radius: 8px;
        padding: 16px;
        font-weight: 500;
    }

    h1, h2, h3 {
        color: var(--text-color);
        font-weight: 700;
        letter-spacing: -0.025em;
    }

    .stProgress > div > div > div > div {
        background: var(--button-bg);
    }

    .header-container {
        position: sticky;
        top: 0;
        background-color: var(--secondary-bg);
        z-index: 1000;
        padding: 1.5rem;
        border-bottom: 2px solid var(--border-color);
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 2px 4px var(--shadow-color);
    }

    .footer-container {
        background-color: var(--secondary-bg);
        padding: 1.5rem;
        text-align: center;
        border-top: 2px solid var(--border-color);
        margin-top: 2rem;
        font-size: 0.9rem;
        color: var(--text-color);
    }

    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 140px;
        background-color: var(--accent-color);
        color: #ffffff;
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -70px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.85rem;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }

    a {
        color: var(--accent-color);
        text-decoration: none;
        font-weight: 500;
    }

    a:hover {
        text-decoration: underline;
        color: #2563eb;
    }

    .stPlotlyChart {
        background-color: var(--chart-bg);
        border-radius: 8px;
        box-shadow: 0 3px 6px var(--shadow-color);
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header with Branding ---
st.markdown("""
    <div class='header-container'>
        <div style='display: flex; align-items: center;'>
            <h2 style='margin: 0; font-size: 1.8rem;'>📰 Advanced News Classifier</h2>
            <span style='margin-left: 1rem; color: var(--text-color); opacity: 0.7; font-size: 0.9rem;'>by Sravan Kodari</span>
        </div>
        <div style='display: flex; gap: 1rem;'>
            <a href='https://github.com/SRAVAN-DSAI/nlp_news_pipeline' target='_blank'>GitHub</a>
            <a href='https://www.linkedin.com/in/sravan-kodari' target='_blank'>LinkedIn</a>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- Enforce Light Mode ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

st.markdown("""
    <script>
        document.body.setAttribute('data-theme', 'light');
    </script>
""", unsafe_allow_html=True)

# --- Configuration Paths ---
LOCAL_MODEL_DIR_RELATIVE = "models/trained_news_classifier"
LOCAL_LABEL_MAP_FILE_RELATIVE = "models/label_map.pkl"

# --- Model Loading ---
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        model_dir_to_load = os.path.join(project_root, LOCAL_MODEL_DIR_RELATIVE)
        label_map_file_to_load = os.path.join(project_root, LOCAL_LABEL_MAP_FILE_RELATIVE)

        loading_container = st.container()
        with loading_container:
            st.markdown("### Model Initialization")
            loading_status = st.empty()
            my_bar = st.progress(0)
            steps = ["Initializing...", "Checking device...", "Loading label map...", "Loading tokenizer...", "Loading model..."]
            for i, step in enumerate(steps):
                loading_status.info(step)
                my_bar.progress((i + 1) * 20)
                time.sleep(0.3)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            loading_status.info(f"Using device: **{device.type.upper()}**")

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
    batch_size = 32  # Optimized for memory efficiency
    for batch_idx in range(0, total_texts, batch_size):
        batch_texts = text_list[batch_idx:batch_idx + batch_size]
        if progress_text_placeholder:
            progress_text_placeholder.text(f"Predicting batch {batch_idx//batch_size + 1}/{((total_texts-1)//batch_size) + 1}...")
        
        inputs = loaded_tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=128).to(current_device)
        with torch.no_grad():
            output = loaded_model(**inputs)
            logits = output.logits

        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_label_indices = torch.argmax(probabilities, dim=-1).cpu().numpy()
        predicted_scores = probabilities[torch.arange(probabilities.size(0)), predicted_label_indices].cpu().numpy()

        for i, text in enumerate(batch_texts):
            if not isinstance(text, str) or not text.strip():
                results.append({"text": text, "predicted_category": "Invalid/Empty Input", "confidence": 0.0, "raw_probabilities": {}})
                continue
            predicted_category = id_to_label_map.get(predicted_label_indices[i], "Unknown")
            raw_probs_dict = {id_to_label_map.get(idx, f"LABEL_{idx}"): prob for idx, prob in enumerate(probabilities[i].cpu().numpy())}
            results.append({
                "text": text,
                "predicted_category": predicted_category,
                "confidence": predicted_scores[i],
                "raw_probabilities": raw_probs_dict
            })
    if progress_text_placeholder:
        progress_text_placeholder.empty()
    return results

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Configuration")
    with st.expander("Prediction Settings", expanded=True):
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.75,
            step=0.05,
            help="Filter predictions by minimum confidence level.",
            key="confidence_threshold"
        )
        batch_size_limit = st.number_input(
            "Max Batch Size",
            min_value=1,
            max_value=500,
            value=100,
            step=10,
            help="Limit the number of articles processed in one batch."
        )
    with st.expander("Visualization Settings"):
        chart_theme = st.selectbox(
            "Chart Theme",
            options=["plotly", "plotly_white", "seaborn", "ggplot2"],
            index=0,
            help="Select the visual theme for charts."
        )
        show_grid = st.checkbox("Show Grid in Charts", value=True)

# --- Session State Initialization ---
if 'text_input_single' not in st.session_state:
    st.session_state.text_input_single = ""
if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 0
if 'batch_results_df' not in st.session_state:
    st.session_state.batch_results_df = pd.DataFrame()
if 'batch_category_counts' not in st.session_state:
    st.session_state.batch_category_counts = pd.Series()
if 'batch_download_data' not in st.session_state:
    st.session_state.batch_download_data = None
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

def clear_all_inputs():
    st.session_state.text_input_single = ""
    st.session_state.file_uploader_key += 1
    st.session_state.batch_results_df = pd.DataFrame()
    st.session_state.batch_category_counts = pd.Series()
    st.session_state.batch_download_data = None
    st.success("All inputs and results cleared!")
    time.sleep(0.5)
    st.rerun()

st.sidebar.button("🧹 Clear All", on_click=clear_all_inputs)

# --- Main Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📝 Single Article",
    "📊 Batch Analysis",
    "🧠 Model Insights",
    "👋 About"
])

with tab1:
    st.header("Analyze a Single Article")
    st.markdown("Classify a single news article with detailed probability visualizations.")

    sample_article = "Google announced a new AI research center in London led by Sundar Pichai, focusing on advanced machine learning solutions."
    
    col_input, col_sample = st.columns([3, 1])
    with col_input:
        user_input_single = st.text_area(
            "Article Text",
            key="text_input_single",
            height=200,
            placeholder="e.g., 'Scientists discover new exoplanet with potential for life.'",
            help="Enter or paste a news article for classification."
        )
        if user_input_single and len(user_input_single.strip()) < 10:
            st.warning("Input is too short. Please provide at least 10 characters.")
    with col_sample:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Try Sample", help="Load a sample article"):
            st.session_state.text_input_single = sample_article

    if st.button("🚀 Classify", help="Run classification on the input text"):
        if user_input_single and len(user_input_single.strip()) >= 10:
            with st.spinner("Classifying article..."):
                single_result = predict_category([user_input_single], model, tokenizer, id_to_label, device)[0]
            
            st.subheader("Classification Result")
            col_result, col_probs = st.columns([1, 2])
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
                    "Probability": single_result['raw_probabilities'].values()
                }).sort_values(by="Probability", ascending=False)
                st.dataframe(prob_df, hide_index=True, use_container_width=True)

                fig = px.bar(
                    prob_df,
                    x="Category",
                    y="Probability",
                    title="Category Probability Distribution",
                    color="Category",
                    color_discrete_sequence=px.colors.qualitative.Plotly,
                    template=chart_theme
                )
                fig.update_layout(
                    yaxis_title="Probability",
                    xaxis_title="Category",
                    showlegend=False,
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=20),
                    yaxis=dict(gridcolor="#e2e8f0" if show_grid else None),
                    xaxis=dict(gridcolor="#e2e8f0" if show_grid else None)
                )
                fig.update_yaxes(range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)

                # Export visualization
                img_buffer = io.BytesIO()
                fig.write_image(img_buffer, format="png")
                st.download_button(
                    label="📥 Download Chart",
                    data=img_buffer,
                    file_name=f"single_article_prob_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    help="Download the probability distribution chart."
                )
        else:
            st.error("Please enter a valid article text (minimum 10 characters).")

with tab2:
    st.header("Batch Analysis")
    st.markdown("Upload a file to classify multiple articles and visualize category distributions.")

    uploaded_file = st.file_uploader(
        "Upload File",
        type=["txt", "csv"],
        key=f"uploader_{st.session_state.file_uploader_key}",
        help="Supported formats: TXT (one article per line) or CSV (with 'text' column)"
    )

    texts_from_file = []
    if uploaded_file:
        try:
            if uploaded_file.type == "text/plain":
                stringio_obj = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                texts_from_file = [line.strip() for line in stringio_obj.readlines() if line.strip()]
                st.info(f"Loaded **{len(texts_from_file)}** articles from TXT.")
            elif uploaded_file.type == "text/csv":
                df_file = pd.read_csv(uploaded_file)
                if 'text' in df_file.columns:
                    texts_from_file = df_file['text'].dropna().tolist()
                    st.info(f"Loaded **{len(texts_from_file)}** articles from CSV.")
                else:
                    st.error("CSV must contain a 'text' column.")
            if len(texts_from_file) > batch_size_limit:
                texts_from_file = texts_from_file[:batch_size_limit]
                st.warning(f"Processing limited to {batch_size_limit} articles per batch.")
        except Exception as e:
            st.error(f"Error processing file: {e}")

    if st.button("🚀 Run Batch Analysis", help="Classify all articles in the uploaded file"):
        if texts_from_file:
            with st.spinner("Processing batch..."):
                progress_text = st.empty()
                progress_bar = st.progress(0)
                all_raw_results = []
                for i, batch in enumerate([texts_from_file[i:i+32] for i in range(0, len(texts_from_file), 32)]):
                    batch_results = predict_category(batch, model, tokenizer, id_to_label, device, progress_text)
                    all_raw_results.extend(batch_results)
                    progress_bar.progress((i + 1) / ((len(texts_from_file) - 1) // 32 + 1))
                progress_text.empty()
                progress_bar.empty()
                st.session_state.all_raw_results = all_raw_results

                results_df = pd.DataFrame([
                    {
                        "Text": res["text"][:150] + "..." if len(res["text"]) > 150 else res["text"],
                        "Category": res["predicted_category"],
                        "Confidence": f"{res['confidence']:.2%}",
                        "Threshold Met": "✅" if res["confidence"] >= confidence_threshold else "❌"
                    } for res in all_raw_results
                ])
                st.session_state.batch_results_df = results_df

                st.subheader("Batch Results")
                st.dataframe(results_df, use_container_width=True, height=400)

                filtered_results = [res for res in all_raw_results if res["confidence"] >= confidence_threshold]
                if filtered_results:
                    category_counts = pd.Series([res["predicted_category"] for res in filtered_results]).value_counts()
                    st.session_state.batch_category_counts = category_counts

                    fig = px.bar(
                        x=category_counts.index,
                        y=category_counts.values,
                        title="Category Distribution",
                        labels={"x": "Category", "y": "Count"},
                        color=category_counts.index,
                        color_discrete_sequence=px.colors.qualitative.Plotly,
                        template=chart_theme
                    )
                    fig.update_layout(
                        showlegend=False,
                        height=400,
                        margin=dict(l=20, r=20, t=50, b=20),
                        yaxis=dict(gridcolor="#e2e8f0" if show_grid else None),
                        xaxis=dict(gridcolor="#e2e8f0" if show_grid else None)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Export visualization
                    img_buffer = io.BytesIO()
                    fig.write_image(img_buffer, format="png")
                    st.download_button(
                        label="📥 Download Chart",
                        data=img_buffer,
                        file_name=f"batch_category_dist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        help="Download the category distribution chart."
                    )

                    csv_buffer = io.StringIO()
                    full_results_df = pd.DataFrame([
                        {
                            "Text": res["text"],
                            "Category": res["predicted_category"],
                            "Confidence": res["confidence"],
                            "Threshold Met": "Yes" if res["confidence"] >= confidence_threshold else "No",
                            **res["raw_probabilities"]
                        } for res in all_raw_results
                    ])
                    full_results_df.to_csv(csv_buffer, index=False)
                    st.session_state.batch_download_data = csv_buffer.getvalue()

                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=st.session_state.batch_download_data,
                        file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="Download detailed results with probabilities."
                    )
                else:
                    st.info("No predictions meet the current confidence threshold.")
        else:
            st.error("Please upload a valid file (TXT or CSV with 'text' column).")

with tab3:
    st.header("🧠 Model Insights")
    st.markdown("Explore the machine learning model and its performance metrics.")

    st.subheader("Model Architecture")
    st.markdown(f"""
        - **Base Model:** DistilBERT (Fine-tuned)
        - **Task:** Multi-class Text Classification
        - **Dataset:** AG News
        - **Categories:** {', '.join(id_to_label.values())}
        - **Device:** {device.type.upper()}
        - **Max Sequence Length:** 128 tokens
        - **Training Details:** Fine-tuned on 120K news articles
    """)

    st.subheader("Performance Metrics")
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "F1-Score", "Precision", "Recall"],
        "Value": [0.9422, 0.9400, 0.9400, 0.9400]
    })
    fig_metrics = px.bar(
        metrics_df,
        x="Metric",
        y="Value",
        title="Model Performance Metrics",
        color="Metric",
        color_discrete_sequence=px.colors.qualitative.Plotly,
        template=chart_theme
    )
    fig_metrics.update_layout(
        yaxis_title="Score",
        xaxis_title="Metric",
        showlegend=False,
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis=dict(gridcolor="#e2e8f0" if show_grid else None, range=[0, 1]),
        xaxis=dict(gridcolor="#e2e8f0" if show_grid else None)
    )
    st.plotly_chart(fig_metrics, use_container_width=True)

    # Export visualization
    img_buffer = io.BytesIO()
    fig_metrics.write_image(img_buffer, format="png")
    st.download_button(
        label="📥 Download Metrics Chart",
        data=img_buffer,
        file_name=f"model_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
        mime="image/png",
        help="Download the performance metrics chart."
    )

with tab4:
    st.header("👋 About")
    st.markdown("""
        This advanced news article classifier demonstrates a robust NLP pipeline using Streamlit, Hugging Face Transformers, and optimized batch processing. 
        It provides interactive visualizations and detailed insights for both single and batch article classification.

        **Connect with me:**
        - **Sravan Kodari**
        - [GitHub](https://github.com/SRAVAN-DSAI/nlp_news_pipeline)
        - [LinkedIn](https://www.linkedin.com/in/sravan-kodari)
        - Email: sravankodari4@gmail.com

        <small>Version 2.0 | Last updated: July 2025</small>
    """, unsafe_allow_html=True)

# --- Footer ---
st.markdown(f"""
    <div class='footer-container'>
        <p>© 2025 Sravan Kodari | Session ID: {st.session_state.session_id[:8]} | 
        <a href='https://github.com/SRAVAN-DSAI/nlp_news_pipeline' target='_blank'>GitHub</a> | 
        <a href='https://www.linkedin.com/in/sravan-kodari' target='_blank'>LinkedIn</a></p>
    </div>
""", unsafe_allow_html=True)

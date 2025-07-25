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
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/SRAVAN-DSAI/nlp_news_pipeline',
        'Report a Bug': 'mailto:sravankodari4@gmail.com',
        'About': 'News Article Classifier by Sravan Kodari'
    }
)

# --- Custom CSS for Enhanced UI Alignment ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        /* Light Mode Variables */
        --primary-bg: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        --secondary-bg: #f8fafc;
        --text-color: #1e293b;
        --accent-color: #3b82f6;
        --button-bg: linear-gradient(45deg, #3b82f6, #1e40af);
        --button-hover-bg: linear-gradient(45deg, #1e40af, #3b82f6);
        --button-text: #ffffff;
        --border-color: #e2e8f0;
        --shadow-color: rgba(0, 0, 0, 0.05);
        --alert-success-bg: #dcfce7;
        --alert-success-text: #166534;
        --alert-warning-bg: #fef3c7;
        --alert-warning-text: #d97706;
        --chart-bg: #ffffff;
        --chart-text: #1e293b;
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
        box-shadow: 2px 0 8px var(--shadow-color);
        padding: 1.5rem;
    }

    .stSidebar .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        padding: 0.75rem;
        background-color: var(--secondary-bg);
        border-radius: 12px;
        box-shadow: 0 2px 6px var(--shadow-color);
    }

    .stTabs [data-baseweb="tab-list"] button {
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        background-color: transparent;
        font-weight: 600;
        color: var(--text-color);
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab-list"] button:hover {
        background-color: var(--border-color);
        transform: translateY(-1px);
    }

    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: var(--accent-color);
        color: var(--button-text);
        box-shadow: 0 2px 4px var(--shadow-color);
    }

    .stButton>button {
        background: var(--button-bg);
        color: var(--button-text);
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px var(--shadow-color);
    }

    .stButton>button:hover {
        background: var(--button-hover-bg);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px var(--shadow-color);
    }

    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid var(--border-color);
        padding: 1rem;
        font-size: 1rem;
        background-color: var(--chart-bg);
        color: var(--text-color);
        transition: border-color 0.2s ease;
    }

    .stTextArea textarea:focus {
        border-color: var(--accent-color);
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }

    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 6px var(--shadow-color);
        background-color: var(--chart-bg);
    }

    .stAlert[data-testid="stAlertSuccess"] {
        background-color: var(--alert-success-bg);
        color: var(--alert-success-text);
        border-radius: 10px;
        padding: 1rem;
    }

    .stAlert[data-testid="stAlertWarning"] {
        background-color: var(--alert-warning-bg);
        color: var(--alert-warning-text);
        border-radius: 10px;
        padding: 1rem;
    }

    h1, h2, h3 {
        color: var(--text-color);
        font-weight: 700;
        margin-bottom: 0.5rem;
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
        border-bottom: 1px solid var(--border-color);
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 2px 4px var(--shadow-color);
    }

    .footer-container {
        background-color: var(--secondary-bg);
        padding: 1.5rem;
        text-align: center;
        border-top: 1px solid var(--border-color);
        margin-top: 2rem;
        font-size: 0.9rem;
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
        color: var(--button-text);
        text-align: center;
        border-radius: 6px;
        padding: 0.5rem;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
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
    }

    .stExpander {
        border-radius: 10px;
        background-color: var(--chart-bg);
        box-shadow: 0 1px 3px var(--shadow-color);
    }

    .stExpander > div > div {
        border: none;
    }

    .card {
        background-color: var(--chart-bg);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 6px var(--shadow-color);
        margin-bottom: 1.5rem;
    }

    .stMetric {
        background-color: var(--chart-bg);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 1px 3px var(--shadow-color);
    }
    </style>
""", unsafe_allow_html=True)

# --- Header with Branding ---
st.markdown("""
    <div class='header-container'>
        <div style='display: flex; align-items: center; gap: 1rem;'>
            <h2 style='margin: 0; font-size: 1.8rem;'>📰 News Classifier</h2>
            <span style='color: var(--text-color); opacity: 0.7; font-size: 1rem;'>by Sravan Kodari</span>
        </div>
        <div style='display: flex; gap: 1.5rem;'>
            <a href='https://github.com/SRAVAN-DSAI/nlp_news_pipeline' target='_blank'>GitHub</a>
            <a href='https://www.linkedin.com/in/sravan-kodari' target='_blank'>LinkedIn</a>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- Theme Setting (Enforced Light Mode) ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Inject JavaScript to enforce light mode
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
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_dir_to_load = os.path.join(project_root, LOCAL_MODEL_DIR_RELATIVE)
    label_map_file_to_load = os.path.join(project_root, LOCAL_LABEL_MAP_FILE_RELATIVE)

    with st.container(border=True):
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

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Settings", divider="gray")
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
    with st.expander("Batch Processing"):
        max_batch_size = st.number_input(
            "Max Batch Size",
            min_value=1,
            max_value=100,
            value=50,
            step=1,
            help="Limit the number of articles processed in one batch."
        )

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

def clear_all_inputs():
    st.session_state.text_input_single = ""
    st.session_state.file_uploader_key += 1
    st.session_state.batch_results_df = pd.DataFrame()
    st.session_state.batch_category_counts = pd.Series()
    st.session_state.batch_download_data = None
    st.success("All inputs and results cleared!")
    time.sleep(0.5)
    st.rerun()

st.sidebar.button("🧹 Clear All", on_click=clear_all_inputs, type="primary")

# --- Main Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📝 Single Article",
    "📊 Batch Analysis",
    "🧠 Model Details",
    "👋 About"
])

with tab1:
    st.header("Analyze a Single Article", divider="gray")
    st.markdown("Classify a single news article and view detailed category predictions.")

    sample_article = "Google announced today that Sundar Pichai will lead a new AI initiative in London. The company's stock rose slightly after the announcement."
    
    with st.container(border=True):
        col_input, col_sample = st.columns([3, 1], vertical_alignment="top")
        with col_input:
            user_input_single = st.text_area(
                "Article Text",
                key="text_input_single",
                height=150,
                placeholder="e.g., 'Scientists discover new exoplanet with potential for life.'",
                help="Paste or type a news article here."
            )
            if user_input_single and len(user_input_single.strip()) < 10:
                st.warning("Input is too short. Please provide more text for accurate classification.")
        with col_sample:
            st.markdown("<br>", unsafe_allow_html=True)
            st.button(
                "Try Sample",
                help="Load a sample article",
                on_click=lambda: st.session_state.update({"text_input_single": sample_article}),
                type="secondary"
            )

        if st.button("🚀 Classify", help="Run classification on the input text", type="primary"):
            if user_input_single and len(user_input_single.strip()) >= 10:
                with st.spinner("Classifying..."):
                    single_result = predict_category([user_input_single], model, tokenizer, id_to_label, device)[0]
                
                st.subheader("Classification Result", divider="gray")
                col_result, col_probs = st.columns([1, 1], gap="medium")
                with col_result:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    if single_result["confidence"] >= confidence_threshold:
                        st.success(f"**Category:** {single_result['predicted_category']}")
                        st.metric("Confidence", f"{single_result['confidence']:.2%}")
                    else:
                        st.warning(f"**Category:** {single_result['predicted_category']} (Below threshold)")
                        st.metric("Confidence", f"{single_result['confidence']:.2%}")
                    st.markdown("</div>", unsafe_allow_html=True)
                with col_probs:
                    prob_df = pd.DataFrame({
                        "Category": single_result['raw_probabilities'].keys(),
                        "Probability": [f"{v:.2%}" for v in single_result['raw_probabilities'].values()]
                    }).sort_values(by="Probability", ascending=False)
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.subheader("Category Probabilities")
                    st.dataframe(prob_df, hide_index=True, use_container_width=True)
                    chart_data = {
                        "type": "bar",
                        "data": {
                            "labels": list(single_result['raw_probabilities'].keys()),
                            "datasets": [{
                                "label": "Probability",
                                "data": list(single_result['raw_probabilities'].values()),
                                "backgroundColor": ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b"],
                                "borderColor": ["#2563eb", "#dc2626", "#16a34a", "#d97706"],
                                "borderWidth": 1
                            }]
                        },
                        "options": {
                            "plugins": {
                                "legend": {"display": False},
                                "tooltip": {"enabled": True},
                                "title": {
                                    "display": True,
                                    "text": "Category Probabilities",
                                    "font": {"size": 16, "family": "'Inter', sans-serif"},
                                    "color": "var(--chart-text)"
                                }
                            },
                            "scales": {
                                "y": {
                                    "beginAtZero": True,
                                    "title": {
                                        "display": True,
                                        "text": "Probability",
                                        "color": "var(--chart-text)",
                                        "font": {"family": "'Inter', sans-serif"}
                                    },
                                    "ticks": {"color": "var(--chart-text)"},
                                    "grid": {"color": "var(--border-color)"}
                                },
                                "x": {
                                    "title": {
                                        "display": True,
                                        "text": "Category",
                                        "color": "var(--chart-text)",
                                        "font": {"family": "'Inter', sans-serif"}
                                    },
                                    "ticks": {"color": "var(--chart-text)"},
                                    "grid": {"color": "var(--border-color)"}
                                }
                            },
                            "backgroundColor": "var(--chart-bg)"
                        }
                    }
                    st.markdown("<div class='tooltip'>Hover over bars for details<span class='tooltiptext'>Click bars to highlight</span></div>", unsafe_allow_html=True)
                    st.markdown(f"```chartjs\n{chart_data}\n```")
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.error("Please enter a valid article text.")

with tab2:
    st.header("Batch Analysis", divider="gray")
    st.markdown("Upload a file to classify multiple articles and analyze category distributions.")

    with st.container(border=True):
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
                    st.info(f"Loaded **{len(texts_from_file)}** articles.")
                elif uploaded_file.type == "text/csv":
                    df_file = pd.read_csv(uploaded_file)
                    if 'text' in df_file.columns:
                        texts_from_file = df_file['text'].tolist()
                        st.info(f"Loaded **{len(texts_from_file)}** articles from CSV.")
                    else:
                        st.error("CSV must have a 'text' column.")
                if len(texts_from_file) > max_batch_size:
                    texts_from_file = texts_from_file[:max_batch_size]
                    st.warning(f"Processing limited to {max_batch_size} articles per batch.")
            except Exception as e:
                st.error(f"Error processing file: {e}")

        if st.button("🚀 Run Batch Analysis", help="Classify all articles in the uploaded file", type="primary"):
            if texts_from_file:
                with st.spinner("Processing batch..."):
                    progress_text = st.empty()
                    all_raw_results = predict_category(texts_from_file, model, tokenizer, id_to_label, device, progress_text)
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

                    st.subheader("Batch Results", divider="gray")
                    st.dataframe(results_df, use_container_width=True, height=400)

                    filtered_results = [res for res in all_raw_results if res["confidence"] >= confidence_threshold]
                    if filtered_results:
                        category_counts = pd.Series([res["predicted_category"] for res in filtered_results]).value_counts()
                        st.session_state.batch_category_counts = category_counts

                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        st.subheader("Category Distribution")
                        chart_data = {
                            "type": "bar",
                            "data": {
                                "labels": category_counts.index.tolist(),
                                "datasets": [{
                                    "label": "Count",
                                    "data": category_counts.values.tolist(),
                                    "backgroundColor": ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b"],
                                    "borderColor": ["#2563eb", "#dc2626", "#16a34a", "#d97706"],
                                    "borderWidth": 1
                                }]
                            },
                            "options": {
                                "plugins": {
                                    "legend": {"display": False},
                                    "tooltip": {"enabled": True},
                                    "title": {
                                        "display": True,
                                        "text": "Category Distribution",
                                        "font": {"size": 16, "family": "'Inter', sans-serif"},
                                        "color": "var(--chart-text)"
                                    }
                                },
                                "scales": {
                                    "y": {
                                        "beginAtZero": True,
                                        "title": {
                                            "display": True,
                                            "text": "Count",
                                            "color": "var(--chart-text)",
                                            "font": {"family": "'Inter', sans-serif"}
                                        },
                                        "ticks": {"color": "var(--chart-text)"},
                                        "grid": {"color": "var(--border-color)"}
                                    },
                                    "x": {
                                        "title": {
                                            "display": True,
                                            "text": "Category",
                                            "color": "var(--chart-text)",
                                            "font": {"family": "'Inter', sans-serif"}
                                        },
                                        "ticks": {"color": "var(--chart-text)"},
                                        "grid": {"color": "var(--border-color)"}
                                    }
                                },
                                "backgroundColor": "var(--chart-bg)"
                            }
                        }
                        st.markdown("<div class='tooltip'>Hover over bars for details<span class='tooltiptext'>Click bars to highlight</span></div>", unsafe_allow_html=True)
                        st.markdown(f"```chartjs\n{chart_data}\n```")
                        st.markdown("</div>", unsafe_allow_html=True)

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
                            file_name="batch_predictions.csv",
                            mime="text/csv",
                            help="Download detailed results with probabilities.",
                            type="primary"
                        )
                    else:
                        st.info("No predictions meet the current confidence threshold.")
            else:
                st.error("Please upload a valid file.")

with tab3:
    st.header("🧠 Model Details", divider="gray")
    st.markdown("Explore the machine learning model behind the news classifier.")

    with st.container(border=True):
        st.subheader("Model Architecture")
        st.markdown(f"""
            - **Base Model:** DistilBERT
            - **Task:** Multi-class Text Classification
            - **Dataset:** AG News
            - **Categories:** {', '.join(id_to_label.values())}
            - **Device:** {device.type.upper()}
        """)

        st.subheader("Performance Metrics")
        col_acc, col_f1, col_precision, col_recall = st.columns(4, gap="medium")
        with col_acc:
            st.metric("Accuracy", "94.22%")
        with col_f1:
            st.metric("F1-Score", "94.00%")
        with col_precision:
            st.metric("Precision", "94.00%")
        with col_recall:
            st.metric("Recall", "94.00%")

with tab4:
    st.header("👋 About", divider="gray")
    with st.container(border=True):
        st.markdown("""
            This app showcases an end-to-end NLP pipeline for news article classification, built with Streamlit, Hugging Face Transformers, and Apache Spark.

            **Connect with me:**
            - **Sravan Kodari**
            - [GitHub](https://github.com/SRAVAN-DSAI/nlp_news_pipeline)
            - [LinkedIn](https://www.linkedin.com/in/sravan-kodari)
            - Email: sravankodari4@gmail.com

            <small>Last updated: July 2025</small>
        """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
    <div class='footer-container'>
        <p>© 2025 Sravan Kodari | 
        <a href='https://github.com/SRAVAN-DSAI/nlp_news_pipeline' target='_blank'>GitHub</a> | 
        <a href='https://www.linkedin.com/in/sravan-kodari' target='_blank'>LinkedIn</a></p>
    </div>
""", unsafe_allow_html=True)

# scripts/streamlit_app.py (Portfolio-Ready Advanced Version - FINAL UPDATED CODE)

import streamlit as st
import torch
import numpy as np
import os
import pickle
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import matplotlib.pyplot as plt # For plotting
import seaborn as sns # For plotting
import io # For handling file downloads
import time # For simulating loading times / progress

# --- Streamlit App Configuration (MUST be the first Streamlit command) ---
st.set_page_config(
    page_title="News Article Classifier Portfolio Demo",
    layout="wide", # Use 'wide' layout for more screen space
    initial_sidebar_state="expanded" # Keep sidebar open by default
)

# --- Custom CSS for a more polished look ---
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6; /* Light gray background */
        color: #333333;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px; /* Add space between tabs */
        /* Ensure tabs take full width and wrap if needed without breaking layout */
        flex-wrap: wrap; 
        justify-content: flex-start; /* Align tabs to start */
    }
    .stTabs [data-baseweb="tab-list"] button {
        padding: 10px 15px; /* Adjust padding for better look */
        border-radius: 8px 8px 0 0; /* Rounded top corners for tabs */
        border-bottom: 3px solid transparent; /* Highlight active tab */
        transition: all 0.2s ease-in-out;
    }
    .stTabs [data-baseweb="tab-list"] button:hover {
        background-color: #e0e0e0;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #ffffff; /* Active tab background */
        border-bottom: 3px solid #4CAF50; /* Active tab underline */
        color: #4CAF50; /* Active tab text color */
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem; /* Larger tab titles */
        font-weight: bold;
        white-space: nowrap; /* Prevent text from wrapping */
        overflow: visible;   /* Ensure text is not clipped */
        display: block;      /* Ensure it takes up necessary space */
        line-height: 1;      /* Ensure consistent line height */
    }
    .stButton>button {
        background-color: #4CAF50; /* Green button */
        color: white;
        border: none; /* Remove default border */
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1rem;
        transition: background-color 0.3s ease;
        cursor: pointer; /* Indicate clickable */
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #cccccc;
        padding: 10px;
    }
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden; /* Ensures rounded corners apply to content */
    }
    .stAlert {
        border-radius: 8px;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50; /* Darker headings */
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* Custom progress bar styling */
    .stProgress > div > div > div > div {
        background-color: #4CAF50; /* Green progress bar */
    }
    </style>
""", unsafe_allow_html=True)


# --- Configuration Paths (relative to project root) ---
LOCAL_MODEL_DIR_RELATIVE = "models/trained_news_classifier"
LOCAL_LABEL_MAP_FILE_RELATIVE = "models/label_map.pkl"

# --- Model Loading (Cached for performance) ---
@st.cache_resource(show_spinner=False) # Hide default spinner, use custom progress bar
def load_model_and_tokenizer():
    """
    Loads the trained model, tokenizer, and label map.
    Assumes files are present in the defined local paths.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_dir_to_load = os.path.join(project_root, LOCAL_MODEL_DIR_RELATIVE)
    label_map_file_to_load = os.path.join(project_root, LOCAL_LABEL_MAP_FILE_RELATIVE)

    # Custom progress bar for loading
    loading_status = st.empty() # Placeholder for dynamic messages
    my_bar = st.progress(0)

    try:
        my_bar.progress(5, text="Initializing...")
        time.sleep(0.1) # Small delay for visual effect

        # Determine device (GPU if available, else CPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loading_status.info(f"Loading model on: **{device.type.upper()}**")
        my_bar.progress(15)
        time.sleep(0.1)

        # Load label map
        if not os.path.exists(label_map_file_to_load):
            loading_status.error(f"Error: Label map file not found at '{label_map_file_to_load}'.")
            st.stop() # Stop app execution
            return None, None, None, None

        with open(label_map_file_to_load, 'rb') as f:
            loaded_label_map = pickle.load(f)
        id_to_label = {v: k for k, v in loaded_label_map.items()}
        loading_status.info(f"Label map loaded. Found **{len(id_to_label)}** categories.")
        my_bar.progress(40)
        time.sleep(0.1)

        # Load model and tokenizer
        if not os.path.exists(model_dir_to_load) or not os.listdir(model_dir_to_load):
            loading_status.error(f"Error: Model directory not found or empty at '{model_dir_to_load}'.")
            st.stop() # Stop app execution
            return None, None, None, None

        tokenizer = AutoTokenizer.from_pretrained(model_dir_to_load)
        loading_status.info("Tokenizer loaded.")
        my_bar.progress(70)
        time.sleep(0.1)

        model = AutoModelForSequenceClassification.from_pretrained(model_dir_to_load)
        model.to(device)
        model.eval() # Set model to evaluation mode
        loading_status.success("Model loaded successfully!")
        my_bar.progress(100)
        time.sleep(0.2) # Final delay for success message to be seen

        loading_status.empty() # Clear the status message
        my_bar.empty() # Clear the progress bar
        return model, tokenizer, id_to_label, device

    except Exception as e:
        loading_status.error(f"Failed to load model or tokenizer: {e}")
        st.exception(e) # Display full exception for debugging
        st.info("Please ensure the model files are valid and compatible with the installed `transformers` and `torch` versions.")
        st.stop() # Stop app execution on critical error
        return None, None, None, None

# Load the model and tokenizer when the script first runs
model, tokenizer, id_to_label, device = load_model_and_tokenizer()

# --- Prediction Function ---
def predict_category(text_list, loaded_model, loaded_tokenizer, id_to_label_map, current_device, progress_text_placeholder=None):
    results = []
    total_texts = len(text_list)
    for i, text in enumerate(text_list):
        if progress_text_placeholder:
            progress_text_placeholder.text(f"Predicting text {i+1} of {total_texts}...")

        if not isinstance(text, str) or not text.strip(): # Handle potential non-string entries or empty strings
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
        progress_text_placeholder.empty() # Clear progress text when done
    return results

# --- Main App Logic ---

st.sidebar.header("⚙️ App Controls")
confidence_threshold = st.sidebar.slider(
    "Minimum Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.75, # Default threshold
    step=0.05,
    help="Only display predictions with confidence above this value."
)
st.sidebar.markdown("---")

# Initialize session state for clearing inputs and storing results
if 'text_input_single' not in st.session_state:
    st.session_state.text_input_single = ""
if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 0 # Unique key for file uploader
if 'batch_results_df' not in st.session_state:
    st.session_state.batch_results_df = pd.DataFrame()
if 'batch_category_counts' not in st.session_state:
    st.session_state.batch_category_counts = pd.Series()
if 'batch_download_data' not in st.session_state:
    st.session_state.batch_download_data = None


# Function to clear all inputs and results
def clear_all_inputs():
    st.session_state.text_input_single = ""
    st.session_state.file_uploader_key += 1 # Increment key to force file uploader reset
    st.session_state.batch_results_df = pd.DataFrame()
    st.session_state.batch_category_counts = pd.Series()
    st.session_state.batch_download_data = None
    st.success("Inputs and results cleared!")
    time.sleep(0.5) # Small delay for message to be seen
    st.experimental_rerun() # Rerun to apply changes


if st.sidebar.button("🧹 Clear All Inputs & Results"):
    clear_all_inputs()


# --- Main Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📝 Single Article", "📊 Batch Prediction", "🧠 Model Info", "👋 About"])

with tab1:
    st.header("Analyze a Single News Article")
    st.markdown("Paste a news article below to get its predicted category and confidence.")

    sample_article = "Google announced today from Mountain View that Sundar Pichai will lead a new AI initiative in London. The company's stock rose slightly after the announcement."
    
    col_input, col_sample = st.columns([4, 1])
    with col_input:
        user_input_single = st.text_area(
            "Enter News Article Text:",
            key="text_input_single", # Key to manage state
            height=250,
            placeholder="e.g., 'Scientists discover new exoplanet with potential for life and water.'"
        )
    with col_sample:
        st.markdown("<br>", unsafe_allow_html=True) # Add some vertical space
        if st.button("Try Sample Article"):
            st.session_state.text_input_single = sample_article
            st.experimental_rerun() # Rerun to update text area

    if st.button("🚀 Predict Category"):
        if user_input_single:
            with st.spinner("Analyzing article..."):
                single_result = predict_category([user_input_single], model, tokenizer, id_to_label, device)[0]
            
            st.subheader("Prediction Result:")
            
            if single_result["confidence"] >= confidence_threshold:
                st.success(f"**Predicted Category:** <span style='font-size: 1.5em; font-weight: bold;'>{single_result['predicted_category']}</span>", unsafe_allow_html=True)
                st.metric(label="Confidence", value=f"{single_result['confidence']:.2%}")
            else:
                st.warning(f"**Prediction below threshold.** Predicted: {single_result['predicted_category']}")
                st.metric(label="Confidence", value=f"{single_result['confidence']:.2%}")
                st.write("Consider lowering the threshold in the sidebar if you want to see this prediction.")

            st.markdown("---")
            st.write("Top 3 Probabilities:")
            prob_df_single = pd.DataFrame({
                "Category": single_result['raw_probabilities'].keys(),
                "Probability": single_result['raw_probabilities'].values()
            }).sort_values(by="Probability", ascending=False).head(3)
            st.dataframe(prob_df_single, hide_index=True)

            with st.expander("See All Probabilities"):
                all_prob_df_single = pd.DataFrame({
                    "Category": single_result['raw_probabilities'].keys(),
                    "Probability": single_result['raw_probabilities'].values()
                }).sort_values(by="Probability", ascending=False)
                st.dataframe(all_prob_df_single, hide_index=True)

        else:
            st.warning("Please enter some text to predict.")

with tab2:
    st.header("Batch Prediction from File")
    st.markdown("Upload a text file (one article per line) or a CSV file (with a 'text' column) for batch classification.")

    uploaded_file = st.file_uploader(
        "Upload your file here:",
        type=["txt", "csv"],
        key=st.session_state.file_uploader_key # Use key for state management
    )

    texts_from_file = []
    if uploaded_file is not None:
        file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type, "filesize": uploaded_file.size}
        st.write(f"File uploaded: **{file_details['filename']}** ({file_details['filesize'] / 1024:.2f} KB)")

        try:
            if uploaded_file.type == "text/plain": # .txt file
                stringio_obj = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                texts_from_file = stringio_obj.readlines()
                texts_from_file = [line.strip() for line in texts_from_file if line.strip()] # Remove empty lines
                st.info(f"Loaded **{len(texts_from_file)}** lines from text file.")
            elif uploaded_file.type == "text/csv": # .csv file
                df_file = pd.read_csv(uploaded_file)
                if 'text' in df_file.columns:
                    texts_from_file = df_file['text'].tolist()
                    st.info(f"Loaded **{len(texts_from_file)}** articles from 'text' column in CSV.")
                else:
                    st.error("CSV file must contain a column named 'text'.")
                    texts_from_file = []
            
            if not texts_from_file:
                st.warning("No valid texts found in the uploaded file.")

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.info("Please ensure your file is correctly formatted (UTF-8 encoding).")
            texts_from_file = [] # Clear texts on error


    if st.button("🚀 Run Batch Prediction"):
        if texts_from_file:
            st.subheader("Batch Prediction Results:")
            prediction_progress_text = st.empty() # Placeholder for progress text
            
            all_raw_results = predict_category(texts_from_file, model, tokenizer, id_to_label, device, prediction_progress_text)
            
            # Store raw results in session state for re-filtering
            st.session_state.all_raw_results = all_raw_results

            # Filter and display results
            results_df = pd.DataFrame([
                {
                    "Text": res["text"][:150] + "..." if len(res["text"]) > 150 else res["text"], # Truncate text for display
                    "Predicted Category": res["predicted_category"],
                    "Confidence": f"{res['confidence']:.4f}",
                    "Threshold Met": "✅ Yes" if res["confidence"] >= confidence_threshold else "❌ No"
                } for res in all_raw_results
            ])
            st.session_state.batch_results_df = results_df # Store for display
            st.dataframe(results_df, use_container_width=True, height=300)

            # Visualize Category Distribution
            st.subheader("Predicted Category Distribution (Above Threshold)")
            filtered_categories = [res["predicted_category"] for res in all_raw_results if res["confidence"] >= confidence_threshold]
            if filtered_categories:
                category_counts = pd.Series(filtered_categories).value_counts()
                st.session_state.batch_category_counts = category_counts # Store for plotting
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax, palette="viridis")
                ax.set_title("Predicted Categories Distribution (Filtered by Confidence)")
                ax.set_xlabel("Category")
                ax.set_ylabel("Count")
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)
            else:
                st.info("No categories to display as no predictions met the current confidence threshold.")
            
            # Downloadable results
            csv_buffer = io.StringIO()
            # Include full text in download, not truncated
            full_results_df_for_download = pd.DataFrame([
                {
                    "Text": res["text"],
                    "Predicted Category": res["predicted_category"],
                    "Confidence": res["confidence"],
                    "Threshold Met": "Yes" if res["confidence"] >= confidence_threshold else "No",
                    **res["raw_probabilities"] # Expand raw probabilities into columns
                } for res in all_raw_results
            ])
            full_results_df_for_download.to_csv(csv_buffer, index=False)
            st.session_state.batch_download_data = csv_buffer.getvalue() # Store for download button
            
            st.download_button(
                label="📥 Download All Predictions (CSV)",
                data=st.session_state.batch_download_data,
                file_name="batch_predictions.csv",
                mime="text/csv",
                help="Download detailed prediction results including raw probabilities."
            )
        else:
            st.warning("Please upload a file or ensure it contains valid texts for batch prediction.")

    # Re-display previous batch results if they exist in session state (e.g., after slider change)
    if not st.session_state.batch_results_df.empty:
        st.subheader("Current Batch Results (Filtered by Slider):")
        # Re-filter based on current slider value
        filtered_results_rerun = [res for res in st.session_state.all_raw_results if res["confidence"] >= confidence_threshold]
        
        if not filtered_results_rerun:
            st.warning(f"No predictions meet the current confidence threshold of {confidence_threshold:.2f}. Try lowering the threshold or check inputs.")

        results_df_rerun = pd.DataFrame([
            {
                "Text": res["text"][:150] + "..." if len(res["text"]) > 150 else res["text"],
                "Predicted Category": res["predicted_category"],
                "Confidence": f"{res['confidence']:.4f}",
                "Threshold Met": "✅ Yes" if res["confidence"] >= confidence_threshold else "❌ No"
            } for res in st.session_state.all_raw_results
        ])
        st.dataframe(results_df_rerun, use_container_width=True, height=300)

        st.subheader("Predicted Category Distribution (Above Current Threshold)")
        if filtered_results_rerun:
            category_counts_rerun = pd.Series([res["predicted_category"] for res in filtered_results_rerun]).value_counts()
            fig_rerun, ax_rerun = plt.subplots(figsize=(10, 6))
            sns.barplot(x=category_counts_rerun.index, y=category_counts_rerun.values, ax=ax_rerun, palette="viridis")
            ax_rerun.set_title("Predicted Categories Distribution (Filtered)")
            ax_rerun.set_xlabel("Category")
            ax_rerun.set_ylabel("Count")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig_rerun)
        else:
            st.info("No categories to display from previous results as no predictions meet the current confidence threshold.")
        
        if st.session_state.batch_download_data: # Only show download button if data exists
            st.download_button(
                label="📥 Download All Predictions (CSV)",
                data=st.session_state.batch_download_data,
                file_name="batch_predictions.csv",
                mime="text/csv",
                help="Download detailed prediction results including raw probabilities."
            )


with tab3:
    st.header("🧠 Model & Performance Info")
    st.markdown("""
        This section provides details about the underlying machine learning model and its performance.
    """)

    st.subheader("Model Architecture")
    st.markdown(f"""
        - **Base Model:** DistilBERT (`{model.config._name_or_path}`)
        - **Task:** Multi-class Text Classification
        - **Fine-tuned On:** AG News Dataset
        - **Number of Categories:** {len(id_to_label)}
        - **Categories:** {', '.join(id_to_label.values())}
        - **Model Device:** {device.type.upper()}
    """)

    st.subheader("Performance Metrics (Validation Set)")
    st.info("""
        The model was evaluated on a held-out validation set (10% of the total dataset).
        These metrics reflect its performance on unseen data.
    """)
    
    # Hardcode metrics from your model_evaluation.py output
    # You can update these after running model_evaluation.py and getting fresh results
    col_acc, col_f1, col_precision, col_recall = st.columns(4)
    with col_acc:
        st.metric(label="Overall Accuracy", value="94.22%") # From your previous output
    with col_f1:
        st.metric(label="Macro F1-Score", value="94.00%") # Approx from your previous output
    with col_precision:
        st.metric(label="Macro Precision", value="94.00%") # Approx from your previous output
    with col_recall:
        st.metric(label="Macro Recall", value="94.00%") # Approx from your previous output

    st.markdown("---")
    st.markdown("""
        For a detailed classification report and confusion matrix, please refer to the
        `model_evaluation.py` script and its outputs in the GitHub repository.
    """)


with tab4:
    st.header("👋 About This Demo & Contact")
    st.markdown("""
        This Streamlit application showcases a News Article Category Classifier,
        demonstrating a full end-to-end NLP pipeline.

        **Pipeline Overview:**
        1.  **Data Ingestion**: News articles are sourced from the AG News dataset.
        2.  **Data Processing (Apache Spark)**: Raw text is cleaned (e.g., lowercasing, removing special characters, URLs, hashtags, mentions) and prepared for model training.
        3.  **Model Training (Hugging Face Transformers)**: A pre-trained DistilBERT model is fine-tuned on the processed news data for multi-class text classification.
        4.  **Model Deployment (Streamlit)**: The trained model is integrated into this interactive web application for live inference.

        **GitHub Repository:**
        Explore the full source code, detailed setup instructions, and the entire data science pipeline on GitHub:
        [https://github.com/SRAVAN-DSAI/nlp_news_pipeline](https://github.com/SRAVAN-DSAI/nlp_news_pipeline)
        

        ---
        **Connect with me:**
        - **Sravan Kodari**
        - [LinkedIn Profile Link] (e.g., `https://www.linkedin.com/in/sravan-kodari`)
        - [Portfolio Website Link] (e.g., `sravn.pp.ua`)
        - Email: `sravankodari4@gmail.com`

        ---
        <small>Last updated: July 2025.</small>
    """, unsafe_allow_html=True)

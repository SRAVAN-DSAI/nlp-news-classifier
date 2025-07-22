\# NLP News Article Classification Pipeline



This repository contains a modular Python pipeline for ingesting, processing, training, evaluating, and deploying a news article classification model. The pipeline uses Apache Spark for data processing and Hugging Face Transformers for model training.



\## Features



\* \*\*Data Ingestion\*\*: Downloads the `ag\_news` dataset from Hugging Face Hub.

\* \*\*Data Processing\*\*: Cleans and prepares text data using Apache Spark, saving it as Parquet files.

\* \*\*Model Training\*\*: Fine-tunes a DistilBERT model for multi-class text classification.

\* \*\*Model Evaluation\*\*: Provides detailed classification metrics, confusion matrix, and error analysis.

\* \*\*Streamlit Application\*\*: An interactive web interface for real-time model predictions.

\* \*\*Modular Design\*\*: Code organized into separate scripts for clarity and and reusability.



\## Project Structure


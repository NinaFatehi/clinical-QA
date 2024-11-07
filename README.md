# Clinical Note Question Answering (QA) Model

This repository contains a machine learning model for performing **question answering (QA)** on clinical notes. The model takes a clinical question and a corresponding clinical note as input and generates an answer based on the note. This tool is designed to assist healthcare professionals by providing quick insights from extensive clinical documents.

## Table of Contents
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)

## Project Overview

This project uses a fine-tuned transformer model based on the `T5` architecture to perform question answering tasks specifically on clinical data. The model has been trained on clinical datasets and can understand and respond to various medical questions based on given context notes.

## Directory Structure

```plaintext
clinical-note-summarization-clean/
├── app/
│   └── streamlit_app.py          # Streamlit app for QA interaction
├── data/
│   ├── preprocess_data.py         # Script for data preprocessing                
├── models/
│   └── clinical_summary/          # Directory for trained model files
├── src/
│   ├── train_model.py             # Model training script
│   ├── evaluate_model.py          # Model evaluation script
│   └── utils.py                   # Utility functions
├── .gitignore                     # Ignore unnecessary files
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation

## Run the Streamlit App

streamlit run app/streamlit_app.py

# Python Data Science Replicability Project

## Overview
This project focuses on replicating Tadesse et al (2019) "Detection of Depression-Related Posts in Reddit Social Media Forum". We do this by performing feature extraction and classification using machine learning techniques. The text data is preprocessed to extract unigram and bigram features, using the Empath (instead of the LIWC) library and also performing an LDA analysis. Classifiers are trained to distinguish between depression-related and non-depression-related posts.

Key functionalities include:
- Text preprocessing
- N-gram (unigram and bigram) feature extraction
- Empath dictionary analysis
- LDA
- Training and evaluating 5 machine learning models with 6 different feature inputs (5x6 ML models)

  
## Features
- **Data Preprocessing**: Includes tokenization, stopword removal, and TF-IDF vectorization.
- **N-Gram Feature Extraction**: Extracts unigram and bigram features with TF-IDF scores.
- **Feature Saving**: Saves feature matrices and labels into CSV files for reusability.
- **Machine Learning**: Implements classifiers (e.g., SVM, Logistic Regression) to detect depression in text.
- **Label Analysis and Debugging**: Validates labels to ensure they are correctly aligned with the data.

## Project Structure

### Folders
- `data/`
  - `feature_extracted_data/`
    - `unigram_features_with_labels.csv`
    - `bigram_features_with_labels.csv`
    - `empath_features_with_labels.csv`
    - `lda_topic_distributions_with_labels.csv`
  - `preprocessed/`
    - `preprocessed_depression_posts/`
    - `preprocessed_breastcancer_posts/`
  - `raw/`
    - `depression_diagnosed_posts/`
    - `breastcancer_diagnosed_posts/`

### Files
- `data_preprocessing.ipynb`
- `feature_analysis.ipynb`
- `feature_extraction.ipynb`
- `model_training.ipynb`
- `README.md`
- `requirements.txt`
## Setup

### Prerequisites
- Python 3.7+
- Recommended: A virtual environment

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/depression-detection-ngram.git
   cd depression-detection-ngram
Install dependencies:
bash
Copy code
pip install -r requirements.txt
Required Libraries
pandas
numpy
scikit-learn
matplotlib
nltk

### Usage
1. Feature Extraction
Run the feature_extraction.ipynb notebook to generate unigram and bigram feature matrices. This notebook will:

Load the text data from the data/preprocessed directories.
Extract unigram and bigram features using TF-IDF. Extract features using the Empath dictionary. Extract features using LDA.
Save the feature matrices and labels as CSV files in data/feature_extracted_data/.

2. Training Machine Learning Models
Use the model_training.ipynb notebook to train machine learning models. This notebook will:

Load the saved CSV files (e.g., unigram_features_with_labels.csv, bigram_features_with_labels.csv).
Train classifiers such as SVM, Logistic Regression, MLP, Random Forest and ADABoost on the extracted features.
Evaluate the models and print metrics (e.g., accuracy, precision, recall).

3. Analysis
Run the feature_analysis.ipynb notebook to:

Analyze the extracted features (e.g., top unigrams and bigrams).


### All workflows are demonstrated in the Jupyter notebooks:

data_preprocessing.ipynb: Preprocess raw text data.
feature_extraction.ipynb: Extract features from text data.
model_training.ipynb: Train and evaluate machine learning models.
Results
Model Accuracy: Achieved XX% accuracy on test data using [model name].
Feature Importance: Identified key unigrams and bigrams associated with depression vs. non-depression posts.
Key Files
unigram_features_with_labels.csv: Contains unigram TF-IDF features and labels.
bigram_features_with_labels.csv: Contains bigram TF-IDF features and labels.
empath_features_with_labels.csv: Contains psycholinguistic features and labels.
lda_topic_distributions_with_labels.csv: Contains LDA topic distributions and labels.


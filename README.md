# Python Data Science Replicability Project

## Overview
This project focuses on replicating Tadesse et al (2019) "Detection of Depression-Related Posts in Reddit Social Media Forum". We do this by performing feature extraction and classification using machine learning techniques. The text data is preprocessed to extract unigram and bigram features, using the Empath (instead of the LIWC) library and also performing an LDA analysis. Classifiers are trained to distinguish between depression-related and non-depression-related posts.

 # Detection of Depression-Related Posts in Reddit (Recreating Tadesse et al.)

This repository demonstrates a pipeline for **detecting depression-related posts in Reddit**—inspired by *Tadesse et al.*’s paper, “Detection of Depression-Related Posts in Reddit Social Media Forum.” The overall workflow includes:

1. **Scraping Raw Data** from Reddit.  
2. **Preprocessing** and cleaning text.  
3. **Extracting Features** (N-grams, Empath categories, LDA topics).  
4. **Analyzing** those features (word clouds, correlation tables, etc.).  
5. **Training and Evaluating** ML models to distinguish depression posts from others.

---

## Repository Structure

```
.
├── data/
│   ├── feature_analysis_output/
│   │   ├── breast_cancer_-_bigrams.png
│   │   ├── breast_cancer_-_unigrams.png
│   │   ├── depression_-_bigrams.png
│   │   ├── depression_-_unigrams.png
│   │   ├── Empath_Correlation_Table.csv
│   │   ├── standard_-_bigrams.png
│   │   └── standard_-_unigrams.png
│   ├── preprocessed_posts/
│   │   ├── breastcancer/
│   │   ├── depression/
│   │   └── standard/
│   ├── reddit_scraped_posts/
│   │   ├── breastcancer/
│   │   ├── depression/
│   │   └── standard/
│   └── data_preprocessing/
│       └── data_preprocessing.py
├── feature_analysis/
│   └── feature_analysis.py
├── feature_extraction/
│   ├── feature_extraction.py
│   └── feature_extraction_func.py
├── model_training/
│   └── model_training.py
└── README.md

```

### Main Folders

- **`reddit_scraped_posts/`**: Raw text posts from Reddit.  
- **`preprocessed_posts/`**: Cleaned/preprocessed text.  
- **`data_preprocessing/`**: Contains `data_preprocessing.py`, the script for text cleaning.  
- **`feature_extraction/`**: Scripts (`feature_extraction.py`, `feature_extraction_func.py`) to extract N-gram, LDA, and Empath features.  
- **`feature_analysis/`**: Contains `feature_analysis.py` for visualizing data (e.g., word clouds, correlation tables).  
- **`feature_analysis_output/`**: Where images (word clouds, correlation plots) and CSVs (e.g. Empath correlation) get saved.  
- **`model_training/`**: Contains `model_training.py` to train/evaluate machine-learning models on extracted features.

---

## Goal: Recreating *Tadesse et al.*’s Approach

*Tadesse et al.* introduced a multi-faceted approach for **detecting depression** in Reddit posts, using linguistic and topic-based features. This repository implements a similar pipeline:

1. **Scrape** real Reddit data, focusing on `r/depression`, plus  control groups (`r/breastcancer`) and other random subreddits (labeled “standard”).  
2. **Clean/Preprocess** the text (remove stopwords, usernames, etc.).  
3. **Extract Features**:
   - **N-grams** via TF-IDF  
   - **Empath** for psycholinguistic features  
   - **LDA** for topic modeling  
4. **Train and Evaluate** classification models (e.g., SVM, Random Forest).  
5. **Analyze** each model’s performance using accuracy, F1, precision, and recall.

---

## Usage

1. **Install Dependencies**  
   - Python 3.9+ recommended.  
   - `pip install -r requirements.txt` or manually install `praw`, `nltk`, `scikit-learn`, `matplotlib`, `pandas`, `empath`, `gensim`, etc.

2. **Scrape Data (Optional)**  
   - If you wish to update or expand the raw data, edit/run the scraping script (not fully shown here) to populate `data/reddit_scraped_posts/` with `.txt` files.

3. **Preprocess**  
   - `python data_preprocessing/data_preprocessing.py`  
   - Produces cleaned `.txt` files in `data/preprocessed_posts/`.

4. **Feature Extraction**  
   - `python feature_extraction/feature_extraction.py`  
   - Generates CSV files (e.g., `unigram_features_with_labels.csv`, `lda_topic_distributions_with_labels.csv`) in `data/feature_extracted_data/` (or wherever configured).

5. **Feature Analysis**  
   - `python feature_analysis/feature_analysis.py`  
   - Saves images (like word clouds) in `data/feature_analysis_output/`.

6. **Model Training**  
   - `python model_training/model_training.py`  
   - Loads one or more CSV feature sets, trains selected ML models, prints and saves evaluation metrics (accuracy, F1, etc.).

---

## References

- **Paper**: M. M. Tadesse, H. Lin, B. Xu, and L. Yang, “Detection of Depression-Related Posts in Reddit Social Media Forum,” *IEEE Access*, vol. 7, pp. 44883–44893, 2019.  
- **NLTK & scikit-learn** used heavily for text processing and classification.  
- **Empath** for psycholinguistic feature extraction.  
- **Gensim** for LDA topic modeling.
---
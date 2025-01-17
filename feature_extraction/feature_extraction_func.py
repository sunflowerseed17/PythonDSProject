import os
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from empath import Empath
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from statsmodels.stats.multitest import multipletests
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from collections import defaultdict


import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Base Class
class FeatureExtractor:
    def __init__(self, documents, labels, output_folder="data/feature_extracted_data"):
        self.documents = documents
        self.labels = labels
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        

    def preprocess_text(self, text):
        """
        Tokenize, lowercase, remove stopwords, and stem.
        """
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        tokens = word_tokenize(text.lower())
        return [stemmer.stem(word) for word in tokens if word.isalpha() and word not in stop_words]

    def save_to_csv(self, data, filename):
        """
        Save data to a CSV file.
        """
        # Construct the full file path
        if not filename.startswith(self.output_folder):
            file_path = os.path.join(self.output_folder, filename)
        else:
            file_path = filename

        # Debugging: Print the path being used
        print(f"Saving file to: {file_path}")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save the file if it doesn't already exist
        if not os.path.exists(file_path):
            data.to_csv(file_path, index=False)
            print(f"Saved to {file_path}.")
        else:
            print(f"File already exists at {file_path}.")

# N-Gram Feature Extractor
class NGramFeatureExtractor(FeatureExtractor):
    def __init__(self, documents, labels, output_folder="data/feature_extracted_data"):
        super().__init__(documents, labels, output_folder)
        self.vectorizer_unigram = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
        self.vectorizer_bigram = TfidfVectorizer(ngram_range=(2, 2), stop_words='english')
        self.vectorizer_combined = TfidfVectorizer(ngram_range=(1, 2), stop_words='english') 
        self.unigram_matrix = None
        self.bigram_matrix = None
        self.unigram_feature_names = None
        self.bigram_feature_names = None
        self.combined_matrix = None
        self.combined_feature_names = None

        os.makedirs(self.output_folder, exist_ok=True)

    def extract_features(self):
        """
        Extract unigram and bigram features using TF-IDF.
        """
        print("Extracting unigrams...")
        self.unigram_matrix = self.vectorizer_unigram.fit_transform(self.documents)
        self.unigram_feature_names = self.vectorizer_unigram.get_feature_names_out()
        print(f"Number of unigram features: {len(self.unigram_feature_names)}")

        print("Extracting bigrams...")
        self.bigram_matrix = self.vectorizer_bigram.fit_transform(self.documents)
        self.bigram_feature_names = self.vectorizer_bigram.get_feature_names_out()
        print(f"Number of bigram features: {len(self.bigram_feature_names)}")

        print("Extracting combined unigrams and bigrams...")
        self.combined_matrix = self.vectorizer_combined.fit_transform(self.documents)
        self.combined_feature_names = self.vectorizer_combined.get_feature_names_out()
        print(f"Number of combined unigram and bigram features: {len(self.combined_feature_names)}")

        return self.unigram_matrix, self.bigram_matrix, self.combined_matrix
    
    def save_features(self):
        """
        Save unigram and bigram features with labels as CSV files.
        """
        unigram_file = os.path.join(self.output_folder, "unigram_features_with_labels.csv")
        bigram_file = os.path.join(self.output_folder, "bigram_features_with_labels.csv")

        if not os.path.exists(unigram_file):
            unigram_df = pd.DataFrame(self.unigram_matrix.toarray(), columns=self.unigram_feature_names)
            unigram_df['label'] = self.labels
            unigram_df.to_csv(unigram_file, index=False)
            print(f"Saved unigram features to {unigram_file}.")
        else:
            print(f"Unigram features file already exists at {unigram_file}.")

        if not os.path.exists(bigram_file):
            bigram_df = pd.DataFrame(self.bigram_matrix.toarray(), columns=self.bigram_feature_names)
            bigram_df['label'] = self.labels
            bigram_df.to_csv(bigram_file, index=False)
            print(f"Saved bigram features to {bigram_file}.")
        else:
            print(f"Bigram features file already exists at {bigram_file}.")

    def get_top_features(self, feature_type="unigram", top_n=10):
        """
        Get the top N most common features for unigrams or bigrams based on TF-IDF scores.
        """
        if feature_type == "unigram":
            tfidf_sums = np.array(self.unigram_matrix.sum(axis=0)).flatten()
            feature_names = self.unigram_feature_names
        elif feature_type == "bigram":
            tfidf_sums = np.array(self.bigram_matrix.sum(axis=0)).flatten()
            feature_names = self.bigram_feature_names
        else:
            raise ValueError("Invalid feature_type. Choose 'unigram' or 'bigram'.")

        top_indices = np.argsort(tfidf_sums)[-top_n:]
        print(f"Top {top_n} Most Common {feature_type.capitalize()} Features:")
        for i in reversed(top_indices):
            print(f"{feature_names[i]}: {tfidf_sums[i]:.4f}")

    def train_model(self, feature_type="unigram"):
        """
        Train a Logistic Regression model using unigrams or bigrams.
        """
        if feature_type == "unigram":
            X = self.unigram_matrix
        elif feature_type == "bigram":
            X = self.bigram_matrix
        else:
            raise ValueError("Invalid feature_type. Choose 'unigram' or 'bigram'.")

        X_train, X_test, y_train, y_test = train_test_split(X, self.labels, test_size=0.2, random_state=42)
        print(f"Training set size: {X_train.shape}")
        print(f"Testing set size: {X_test.shape}")

        param_grid = {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['liblinear']
        }
        grid_search = GridSearchCV(LogisticRegression(max_iter=500), param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        print("\nBest Hyperparameters:")
        print(grid_search.best_params_)

        classifier = grid_search.best_estimator_
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

        return classifier

    def compute_frequencies(self, feature_type="unigram"):
        """
        Compute frequencies of unigrams or bigrams for depression and non-depression posts.
        """
        if feature_type == "unigram":
            matrix = self.unigram_matrix
            feature_names = self.unigram_feature_names
        elif feature_type == "bigram":
            matrix = self.bigram_matrix
            feature_names = self.bigram_feature_names
        else:
            raise ValueError("Invalid feature_type. Choose 'unigram' or 'bigram'.")

        depression_indices = [i for i, label in enumerate(self.labels) if label == 1]
        non_depression_indices = [i for i, label in enumerate(self.labels) if label == 0]

        depression_matrix = matrix[depression_indices]
        non_depression_matrix = matrix[non_depression_indices]

        depression_sums = depression_matrix.sum(axis=0).A1
        non_depression_sums = non_depression_matrix.sum(axis=0).A1

        depression_freqs = {feature_names[i]: depression_sums[i] for i in range(len(feature_names))}
        non_depression_freqs = {feature_names[i]: non_depression_sums[i] for i in range(len(feature_names))}

        return depression_freqs, non_depression_freqs

    def get_top_n_features(self, frequencies, top_n=100):
        """
        Get the top N most frequent features from the computed frequencies.
        """
        sorted_features = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:top_n]

    def generate_wordclouds(self):
        """
        Generate 4 word clouds:
        - Unigrams for depression
        - Bigrams for depression
        - Unigrams for non-depression
        - Bigrams for non-depression
        """
        # Compute frequencies for depression and non-depression for unigrams and bigrams
        depression_unigrams, non_depression_unigrams = self.compute_frequencies(feature_type="unigram")
        depression_bigrams, non_depression_bigrams = self.compute_frequencies(feature_type="bigram")

        # Combine word clouds into a dictionary for iteration
        wordcloud_data = {
            "Depression - Unigrams": depression_unigrams,
            "Depression - Bigrams": depression_bigrams,
            "Non-Depression - Unigrams": non_depression_unigrams,
            "Non-Depression - Bigrams": non_depression_bigrams
        }

        # Generate and display each word cloud
        for title, frequencies in wordcloud_data.items():
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(frequencies)
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(title, fontsize=16)
            plt.show()

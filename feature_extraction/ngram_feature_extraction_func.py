###############################################################################
#  IMPORTS
###############################################################################
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from math import log
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from feature_extraction.base_feature_extraction_func import FeatureExtractor
import logging

###############################################################################
# CONFIGURATION CONSTANTS
###############################################################################
UNIGRAM_FEATURES_FILENAME = "unigram_features_with_labels.csv"
BIGRAM_FEATURES_FILENAME = "bigram_features_with_labels.csv"
DEFAULT_PMI_THRESHOLD = 2.0

# WordCloud settings
WORDCLOUD_WIDTH = 800
WORDCLOUD_HEIGHT = 400
WORDCLOUD_BG_COLOR = 'white'
WORDCLOUD_DPI = 300
WORDCLOUD_FIG_SIZE = (10, 6)
WORDCLOUD_TITLE_FONT_SIZE = 16

###############################################################################
# LOGGER CONFIGURATION
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

###############################################################################
# N-GRAM FEATURE EXTRACTOR
###############################################################################
class NGramFeatureExtractor(FeatureExtractor):
    def __init__(self, *, output_folder="data/feature_extracted_data", folders=None):
        super().__init__(folders=folders, output_folder=output_folder)
        self.vectorizer_unigram = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', max_features=5000)
        self.vectorizer_bigram = TfidfVectorizer(ngram_range=(2, 2), stop_words='english', max_features=5000)
        self.vectorizer_combined = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=5000)
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
        logger.info("Extracting unigrams...")
        self.unigram_matrix = self.vectorizer_unigram.fit_transform(self.documents)
        self.unigram_feature_names = self.vectorizer_unigram.get_feature_names_out()
        logger.info("Number of unigram features: %d", len(self.unigram_feature_names))

        logger.info("Extracting bigrams...")
        self.bigram_matrix = self.vectorizer_bigram.fit_transform(self.documents)
        self.bigram_feature_names = self.vectorizer_bigram.get_feature_names_out()
        logger.info("Number of bigram features: %d", len(self.bigram_feature_names))

        logger.info("Extracting combined unigrams and bigrams...")
        self.combined_matrix = self.vectorizer_combined.fit_transform(self.documents)
        self.combined_feature_names = self.vectorizer_combined.get_feature_names_out()
        logger.info("Number of combined unigram and bigram features: %d", len(self.combined_feature_names))

        return self.unigram_matrix, self.bigram_matrix, self.combined_matrix

    def compute_pmi(self, bigram, unigram_counts, total_bigrams, total_unigrams):
        word1, word2 = bigram.split()
        p_bigram = total_bigrams[bigram] / sum(total_bigrams.values())
        p_word1 = unigram_counts[word1] / sum(total_unigrams.values())
        p_word2 = unigram_counts[word2] / sum(total_unigrams.values())
        return log(p_bigram / (p_word1 * p_word2), 2)

    def filter_bigrams_by_pmi(self, threshold=DEFAULT_PMI_THRESHOLD):
        unigram_counts = Counter(word for doc in self.documents for word in doc.split())
        bigram_counts = Counter(
            f"{doc[i]} {doc[i+1]}"  # Added a space between words for clarity
            for doc in [d.split() for d in self.documents]
            for i in range(len(doc) - 1)
        )
        valid_bigrams = {
            bigram: self.compute_pmi(bigram, unigram_counts, bigram_counts, unigram_counts)
            for bigram in self.bigram_feature_names
            if bigram in bigram_counts
        }
        self.bigram_feature_names = [
            bigram for bigram, pmi in valid_bigrams.items() if pmi >= threshold
        ]
        bigram_indices = [
            i for i, bigram in enumerate(self.vectorizer_bigram.get_feature_names_out())
            if bigram in self.bigram_feature_names
        ]
        self.bigram_matrix = self.bigram_matrix[:, bigram_indices]
    
    def save_features(self):
        """
        Save unigram and bigram features with labels as CSV files.
        """
        unigram_file = os.path.join(self.output_folder, UNIGRAM_FEATURES_FILENAME)
        bigram_file = os.path.join(self.output_folder, BIGRAM_FEATURES_FILENAME)

        if not os.path.exists(unigram_file):
            unigram_df = pd.DataFrame(self.unigram_matrix.toarray(), columns=self.unigram_feature_names)
            unigram_df['label'] = self.labels
            unigram_df.to_csv(unigram_file, index=False)
            logger.info("Saved unigram features to %s.", unigram_file)
        else:
            logger.info("Unigram features file already exists at %s.", unigram_file)

        if not os.path.exists(bigram_file):
            bigram_df = pd.DataFrame(self.bigram_matrix.toarray(), columns=self.bigram_feature_names)
            bigram_df['label'] = self.labels
            bigram_df.to_csv(bigram_file, index=False)
            logger.info("Saved bigram features to %s.", bigram_file)
        else:
            logger.info("Bigram features file already exists at %s.", bigram_file)

    def compute_frequencies(self, feature_type="unigram"):
        """
        Compute frequencies of unigrams or bigrams for depression, breastcancer, and standard posts.

        Parameters:
            feature_type (str): Specify 'unigram' or 'bigram' to compute respective frequencies.

        Returns:
            tuple: Three dictionaries containing frequencies for depression, breastcancer, and standard posts.
        """
        if feature_type == "unigram":
            matrix = self.unigram_matrix
            feature_names = self.unigram_feature_names
        elif feature_type == "bigram":
            matrix = self.bigram_matrix
            feature_names = self.bigram_feature_names
        else:
            raise ValueError("Invalid feature_type. Choose 'unigram' or 'bigram'.")

        # Separate indices for depression, breastcancer, and standard posts
        depression_indices = [i for i, label in enumerate(self.labels) if label == 1]
        breastcancer_indices = [i for i, label in enumerate(self.labels) if label == 2]
        standard_indices = [i for i, label in enumerate(self.labels) if label == 0]

        # Subset the matrix for each category
        depression_matrix = matrix[depression_indices]
        breastcancer_matrix = matrix[breastcancer_indices]
        standard_matrix = matrix[standard_indices]

        # Sum the frequencies for each feature
        depression_sums = depression_matrix.sum(axis=0).A1
        breastcancer_sums = breastcancer_matrix.sum(axis=0).A1
        standard_sums = standard_matrix.sum(axis=0).A1

        # Create frequency dictionaries
        depression_freqs = {feature_names[i]: depression_sums[i] for i in range(len(feature_names))}
        breastcancer_freqs = {feature_names[i]: breastcancer_sums[i] for i in range(len(feature_names))}
        standard_freqs = {feature_names[i]: standard_sums[i] for i in range(len(feature_names))}

        return depression_freqs, breastcancer_freqs, standard_freqs
    
    def generate_wordclouds(self):
        """
        Generate word clouds for depression, breastcancer, and standard groups based on unigrams and bigrams.
        """
        # Get unigram and bigram frequencies for all groups
        depression_unigrams, breastcancer_unigrams, standard_unigrams = self.compute_frequencies(feature_type="unigram")
        depression_bigrams, breastcancer_bigrams, standard_bigrams = self.compute_frequencies(feature_type="bigram")

        # Prepare word cloud data for each group and n-gram type
        wordcloud_data = {
            "Depression - Unigrams": depression_unigrams,
            "Breast Cancer - Unigrams": breastcancer_unigrams,
            "Standard - Unigrams": standard_unigrams,
            "Depression - Bigrams": depression_bigrams,
            "Breast Cancer - Bigrams": breastcancer_bigrams,
            "Standard - Bigrams": standard_bigrams,
        }

        # Dictionary to store generated word clouds
        generated_wordclouds = {}

        for title, frequencies in wordcloud_data.items():
            # Clean up frequencies by removing invalid or zero entries
            cleaned_frequencies = {word: freq for word, freq in frequencies.items() if not pd.isna(freq) and freq > 0}
            if not cleaned_frequencies:
                logger.warning("Skipping %s due to empty or invalid frequency data.", title)
                continue

            # Generate word cloud
            wordcloud = WordCloud(
                width=WORDCLOUD_WIDTH,
                height=WORDCLOUD_HEIGHT,
                background_color=WORDCLOUD_BG_COLOR
            ).generate_from_frequencies(cleaned_frequencies)
            generated_wordclouds[title] = wordcloud  # Store the word cloud object

        return generated_wordclouds
    
    def save_wordclouds(self, output_folder):
        wordcloud_data = self.generate_wordclouds()
        os.makedirs(output_folder, exist_ok=True)
        for title, wordcloud in wordcloud_data.items():
            file_name = f"{title.replace(' ', '_').lower()}.png"
            file_path = os.path.join(output_folder, file_name)

            # Save the word cloud image
            plt.figure(figsize=WORDCLOUD_FIG_SIZE)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(title, fontsize=WORDCLOUD_TITLE_FONT_SIZE)
            plt.savefig(file_path, bbox_inches='tight', dpi=WORDCLOUD_DPI)
            plt.close()  # Close the figure to avoid overlapping
            logger.info("Saved word cloud: %s", file_path)
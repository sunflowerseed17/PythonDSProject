###############################################################################
#  IMPORTS
###############################################################################
import os
import numpy as np
import pandas as pd
import logging
from gensim import corpora
from gensim.models import LdaModel
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.conftest import threadpool_limits
import matplotlib.pyplot as plt
from feature_extraction.base_feature_extraction_func import FeatureExtractor

###############################################################################
# CONFIGURATION CONSTANTS
###############################################################################
DEFAULT_NUM_TOPICS = 70
DEFAULT_PASSES = 15
DEFAULT_RANDOM_STATE = 42
DEFAULT_OUTPUT_FOLDER = os.path.join("data", "feature_extracted_data")

LDA_TOPIC_DISTRIBUTIONS_FILENAME = "lda_topic_distributions_with_labels.csv"
DEFAULT_TOPIC_TABLE_OUTPUT = os.path.join("outputs", "topic_table_depressed.png")
DEFAULT_TSNE_OUTPUT = os.path.join("outputs", "tsne_with_categories.png")
DEFAULT_DPI = 300

TSNE_N_COMPONENTS = 2

###############################################################################
# LOGGER CONFIGURATION
###############################################################################
logging.basicConfig(
    level=logging.INFO,  # Adjust level as needed
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

###############################################################################
#  LDA FEATURE EXTRACTOR
###############################################################################
 
class LDAFeatureExtractor(FeatureExtractor):
    def __init__(self, *, num_topics=DEFAULT_NUM_TOPICS, passes=DEFAULT_PASSES, 
                 output_folder=DEFAULT_OUTPUT_FOLDER, random_state=DEFAULT_RANDOM_STATE, folders=None):
        super().__init__(folders=folders, output_folder=output_folder)
        self.num_topics = num_topics
        self.passes = passes
        self.random_state = random_state
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
        self.topic_distributions = None
        self.tsne_results = None

    def preprocess_documents(self, subset_documents=None):
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        documents_to_process = subset_documents if subset_documents is not None else self.documents

        processed_docs = [
            [
                stemmer.stem(word) for word in word_tokenize(doc.lower())
                if word.isalpha() and word not in stop_words
            ]
            for doc in documents_to_process
        ]
        return processed_docs

    def train_lda(self, processed_docs):
        if not processed_docs:
            raise ValueError("Error: No documents provided for LDA training.")
        try:
            self.dictionary = corpora.Dictionary(processed_docs)
            self.corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]
            self.lda_model = LdaModel(
                self.corpus,
                num_topics=self.num_topics,
                id2word=self.dictionary,
                passes=self.passes,
                random_state=self.random_state
            )
        except ValueError as e:
            logger.error("ValueError during LDA training: %s", e, exc_info=True)
            raise
        except Exception as e:
            logger.error("Unexpected error during LDA training: %s", e, exc_info=True)
            raise

    def extract_topic_distributions(self):
        """
        Extract topic distributions for each document.
        """
        self.topic_distributions = [
            dict(self.lda_model.get_document_topics(doc, minimum_probability=0))
            for doc in self.corpus
        ]

    def topic_distribution_to_matrix(self):
        """
        Convert topic distributions to a matrix format.
        """
        matrix = np.zeros((len(self.topic_distributions), self.num_topics))
        for i, distribution in enumerate(self.topic_distributions):
            for topic_id, prob in distribution.items():
                matrix[i, topic_id] = prob
        return matrix

    def save_features(self):
        """
        Save LDA topic distributions with labels to a CSV file.
        """
        if not self.topic_distributions:
            raise ValueError("Topic distributions are not extracted.")

        topic_matrix = self.topic_distribution_to_matrix()
        labels_array = np.array(self.labels)
        lda_features_df = pd.DataFrame(topic_matrix)
        lda_features_df['label'] = labels_array

        lda_features_file = os.path.join(self.output_folder, LDA_TOPIC_DISTRIBUTIONS_FILENAME)
        self.save_to_csv(lda_features_df, lda_features_file)
        logger.info("LDA topic distributions saved to %s", lda_features_file)

    def run_tsne(self):
        if self.topic_distributions is None:
            raise ValueError("Error: Topic distributions are not computed. Run `extract_topic_distributions()` first.")
        try:
            topic_matrix = self.topic_distribution_to_matrix()
            with threadpool_limits(limits=1, user_api="blas"):
                tsne = TSNE(n_components=TSNE_N_COMPONENTS, random_state=self.random_state)
                tsne_results = tsne.fit_transform(topic_matrix)
            return tsne_results
        except MemoryError:
            logger.error("t-SNE computation exceeded memory limits. Try reducing the dataset size.", exc_info=True)
            raise
        except ValueError as e:
            logger.error("ValueError during t-SNE computation: %s", e, exc_info=True)
            raise
        except Exception as e:
            logger.error("Unexpected error during t-SNE computation: %s", e, exc_info=True)
            raise

    def generate_topic_table(self, output_file=DEFAULT_TOPIC_TABLE_OUTPUT):
        """
        Generate a table summarizing LDA topics with most representative words for depressed posts.
        """
        topics = []
        for i, topic in enumerate(self.lda_model.print_topics(num_topics=self.num_topics, num_words=10)):
            topic_id = f"Topic {i + 1}"
            words = ", ".join([word.split("*")[1].replace('"', "").strip() for word in topic[1].split("+")])
            topics.append((topic_id, words))

        topic_df = pd.DataFrame(topics, columns=["Topics", "Most Representative Words"])
        # Dynamic figure height based on number of topics
        fig, ax = plt.subplots(figsize=(12, len(topics) * 0.5))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(
            cellText=topic_df.values,
            colLabels=topic_df.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(topic_df.columns))))

        for key, cell in table.get_celld().items():
            row, col = key
            if row == 0: 
                cell.set_text_props(weight="bold")
                cell.set_fontsize(12)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, bbox_inches="tight", dpi=DEFAULT_DPI)
        logger.info("Topic table for depressed posts saved to %s", output_file)
        plt.show()

    def visualize_tsne(self, tsne_results, topic_matrix, num_topics, topic_categories, 
                       output_file=DEFAULT_TSNE_OUTPUT):
        """
        Visualize t-SNE results with overarching category labels for each cluster.
        """
        if tsne_results is None or topic_matrix is None:
            raise ValueError("t-SNE results and topic matrix must be provided.")

        # Cluster the topic matrix to group points by topic
        clusters = KMeans(n_clusters=num_topics, random_state=self.random_state).fit_predict(topic_matrix)

        plt.figure(figsize=(12, 10))

        for i in range(num_topics):
            indices = np.where(clusters == i)
            plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=f"Topic {i + 1}", alpha=0.6)

            # Add annotations for each cluster with overarching categories
            cluster_center = tsne_results[indices].mean(axis=0)
            category = topic_categories.get(i, f"Topic {i + 1}")  # Fallback if no category is defined
            plt.text(
                cluster_center[0],
                cluster_center[1],
                category,
                fontsize=10,
                ha="center",
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.6),
            )

        plt.title("t-SNE Visualization of LDA Topics with Overarching Categories")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.legend(loc="best", bbox_to_anchor=(1.05, 1), fontsize=9)
        plt.grid(True)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, bbox_inches="tight", dpi=DEFAULT_DPI)
        logger.info("t-SNE visualization with categories saved to %s", output_file)
        plt.show()

    def run_feature_extraction(self):
        """
        Pipeline for feature extraction: preprocess, train LDA, and save features to CSV.
        """
        processed_docs = self.preprocess_documents()
        self.train_lda(processed_docs)
        self.extract_topic_distributions()
        self.save_features()
        logger.info("Feature extraction pipeline complete.")

    def run_feature_analysis(self):
        """
        Pipeline for feature analysis: preprocess documents for depressed posts, train LDA, generate t-SNE visualization with overarching categories, and create topic table.
        """
        # Filter documents for depression label (assuming label 1 corresponds to depressed posts)
        depressed_indices = [i for i, label in enumerate(self.labels) if label == 1]
        depressed_docs = [self.documents[i] for i in depressed_indices]
        processed_docs = self.preprocess_documents(depressed_docs)
        self.train_lda(processed_docs)

        # Extract topic distributions and compute topic matrix
        self.extract_topic_distributions()
        self.topic_matrix = self.topic_distribution_to_matrix()

        # Apply t-SNE for dimensionality reduction
        self.tsne_results = self.run_tsne()

        # Overarching categories for topics (example mapping)
        topic_categories = {
            0: "Parenting and Family",
            1: "Self-Improvement and Values",
            2: "Finance and Banking",
            3: "Planning and Relationships",
            4: "Marriage and Family",
            5: "Mental Health",
            6: "Reflection and Motivation",
            7: "Wishes and Connections",
            8: "Curiosity and Interests",
            9: "Fitness and Health",
            10: "Help and Processes",
            11: "Danish",
            12: "Appearance and Perception",
            13: "Action and Media",
            14: "Life and Time",
            15: "Time and Reflection",
            16: "Jobs and Tasks",
            17: "Anger and Love",
            18: "Faith and Belief",
            19: "Depression and Recovery",
        }

        # Visualize t-SNE with categories
        self.visualize_tsne(
            tsne_results=self.tsne_results,
            topic_matrix=self.topic_matrix,
            num_topics=self.num_topics,
            topic_categories=topic_categories,
            output_file=DEFAULT_TSNE_OUTPUT
        )

        # Generate a topic table summarizing the top words for each topic
        self.generate_topic_table(output_file=DEFAULT_TOPIC_TABLE_OUTPUT)

        logger.info("Feature analysis pipeline for depressed posts complete.")
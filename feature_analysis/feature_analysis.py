import os
import sys
from feature_analysis_func import NGramFeatureAnalyzer, LDAFeatureAnalyzer
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../feature_extraction"))
from feature_extraction_func import NGramFeatureExtractor, LDAFeatureExtractor



def load_documents_and_labels(preprocessed_folder):
    documents = []
    labels = []
    for label, subfolder in enumerate(["depression", "standard"]):  # Adjust folder names if necessary
        folder_path = os.path.join(preprocessed_folder, subfolder)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    documents.append(file.read())
                    labels.append(label)
    return documents, labels

# Example usage
preprocessed_folder = "data/preprocessed_posts"
documents, labels = load_documents_and_labels(preprocessed_folder)
print(f"Loaded {len(documents)} documents with labels: {set(labels)}")

# Initialize the N-gram feature extractor
ngram_extractor = NGramFeatureExtractor(documents, labels)
ngram_extractor.extract_features()

# Get unigram and bigram frequencies
unigram_freqs, bigram_freqs = ngram_extractor.compute_frequencies(feature_type="unigram"), ngram_extractor.compute_frequencies(feature_type="bigram")

# Initialize the LDA feature extractor
lda_extractor = LDAFeatureExtractor(documents, labels, num_topics=20, passes=15)
lda_extractor.run_pipeline()

# Extract LDA model and topic matrix
lda_model = lda_extractor.lda_model
topic_matrix = lda_extractor.topic_distribution_to_matrix()
# Assuming you have already loaded unigram and bigram frequencies, LDA model, topic matrix, and labels.
ngram_analyzer = NGramFeatureAnalyzer(unigram_freqs, bigram_freqs)
ngram_analyzer.visualize_wordclouds()

lda_analyzer = LDAFeatureAnalyzer(lda_model, topic_matrix, labels)
tsne_results = lda_analyzer.run_tsne(perplexity=50, n_iter=500)
cluster_labels = lda_analyzer.cluster_topics(n_clusters=10)
topic_words = lda_analyzer.get_topic_words(top_n=5)
lda_analyzer.visualize_tsne_clusters(tsne_results, cluster_labels, topic_words)

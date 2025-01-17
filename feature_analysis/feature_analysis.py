import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from feature_analysis_func import EmpathFeatureAnalyzer, NGramFeatureAnalyzer, LDAFeatureAnalyzer

# Initialize and run Empath feature analysis
print("Running Empath Feature Analysis...")
empath_analyzer = EmpathFeatureAnalyzer()
empath_analyzer.extract_empath_features()
empath_analyzer.analyze_correlation()
print("Empath Analysis Results:")
print(empath_analyzer.correlation_results.head())

# Initialize and run N-Gram feature analysis
print("\nRunning N-Gram Feature Analysis...")
ngram_analyzer = NGramFeatureAnalyzer()
ngram_analyzer.extract_ngrams()
ngram_analyzer.generate_wordclouds()

<<<<<<< Updated upstream
def load_documents_and_labels(preprocessed_folder):
    documents = []
    labels = []
    for label, subfolder in enumerate(["depression", "standard", "breastcancer"]): 
        folder_path = os.path.join(preprocessed_folder, subfolder)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    documents.append(file.read())
                    labels.append(label)
    return documents, labels
=======
# Initialize and run LDA analysis
print("\nRunning LDA Analysis...")
>>>>>>> Stashed changes

# Mock LDA data for demonstration purposes
lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
topic_matrix = lda_model.fit_transform(np.random.rand(len(ngram_analyzer.documents), 100))  # Random topic matrix for demo

lda_analyzer = LDAFeatureAnalyzer(lda_model, topic_matrix)
tsne_results = lda_analyzer.run_tsne()
cluster_labels = lda_analyzer.cluster_topics()

# Print t-SNE and clustering results
print("t-SNE and Clustering Results:")
print("t-SNE Results Shape:", tsne_results.shape)
print("Cluster Labels:", np.unique(cluster_labels))
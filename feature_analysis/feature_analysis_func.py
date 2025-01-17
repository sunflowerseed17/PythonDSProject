import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd


class NGramFeatureAnalyzer:
    def __init__(self, unigram_freqs, bigram_freqs):
        self.unigram_freqs = unigram_freqs
        self.bigram_freqs = bigram_freqs

    def get_top_n_features(self, frequencies, top_n=10):
        sorted_features = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:top_n]

    def visualize_wordcloud(self, frequencies, title):
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(frequencies)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(title, fontsize=16)
        plt.show()

    def visualize_wordclouds(self):
        self.visualize_wordcloud(self.unigram_freqs, "Unigram Word Cloud")
        self.visualize_wordcloud(self.bigram_freqs, "Bigram Word Cloud")


class LDAFeatureAnalyzer:
    def __init__(self, lda_model, topic_matrix, labels):
        self.lda_model = lda_model
        self.topic_matrix = topic_matrix
        self.labels = labels

    def run_tsne(self, perplexity=30, n_iter=500, random_state=42):
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state)
        tsne_results = tsne.fit_transform(self.topic_matrix)
        return tsne_results

    def cluster_topics(self, n_clusters=10):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(self.topic_matrix)
        return cluster_labels

    def get_topic_words(self, top_n=5):
        topic_words = []
        for topic_id in range(self.topic_matrix.shape[1]):
            words = self.lda_model.show_topic(topic_id, topn=top_n)
            topic_words.append(" ".join([word for word, _ in words]))
        return topic_words

    def visualize_tsne_clusters(self, tsne_results, cluster_labels, topic_words, title="t-SNE Visualization of LDA Topics"):
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            tsne_results[:, 0], tsne_results[:, 1], c=cluster_labels, cmap="viridis", s=30, alpha=0.7
        )
        plt.colorbar(scatter, label="Cluster")
        plt.title(title, fontsize=16)

        for i, (x, y) in enumerate(tsne_results):
            plt.text(x, y, topic_words[cluster_labels[i]], fontsize=8, alpha=0.6)

        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.show()


class EmpathFeatureAnalyzer:
    def __init__(self, correlation_results, categories):
        self.correlation_results = correlation_results
        self.categories = categories
        self.category_correlations = {}

    def group_correlations_by_subcategory(self):
        if self.correlation_results is None:
            raise ValueError("Correlation analysis must be performed before grouping.")

        grouped_results = defaultdict(list)

        for feature, correlation, p_value in zip(
            self.correlation_results['Feature'],
            self.correlation_results['Correlation'],
            self.correlation_results['P-Value']
        ):
            for category, subcategories in self.categories.items():
                if isinstance(subcategories, dict):  # Handle nested subcategories
                    for subcategory, sub_features in subcategories.items():
                        if feature in sub_features:
                            grouped_results[f"{category} - {subcategory}"].append((feature, correlation, p_value))
                elif isinstance(subcategories, list):  # Handle flat categories
                    if feature in subcategories:
                        grouped_results[category].append((feature, correlation, p_value))

        self.category_correlations = {}
        for group, correlations in grouped_results.items():
            avg_correlation = np.mean([c[1] for c in correlations])
            sorted_features = sorted(correlations, key=lambda x: abs(x[1]), reverse=True)
            top_feature = sorted_features[0]
            self.category_correlations[group] = {
                "Example Word": top_feature[0],
                "Correlation": avg_correlation,
                "P-Value": top_feature[2]
            }

    def generate_summary_table(self):
        if not self.category_correlations:
            raise ValueError("Category correlations must be computed before generating a summary table.")

        summary_data = []
        for category, details in self.category_correlations.items():
            summary_data.append({
                "LIWC Category": category,
                "Example Word": details["Example Word"],
                "Correlation": f"{details['Correlation']:.2f}",
                "P-Value": f"{details['P-Value']:.3f}"
            })

        return pd.DataFrame(summary_data)

    def visualize_summary_table(self):
        summary_table = self.generate_summary_table()
        print(summary_table.to_string(index=False))

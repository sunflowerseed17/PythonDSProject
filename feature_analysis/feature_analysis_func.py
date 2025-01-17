
# N=grams Feature Analyzer

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


# Empath Feature Analyzer
class EmpathFeatureAnalyzer(EmpathFeatureExtractor):
    def __init__(self, documents, labels, categories, output_folder="data/feature_extracted_data"):
        super().__init__(documents, labels, categories, output_folder)
        self.category_correlations = {}

    def group_correlations_by_subcategory(self):
        """
        Group feature correlations by subcategories and calculate the average correlation for each group.
        """
        if self.correlation_results is None:
            raise ValueError("Correlation analysis must be performed before grouping.")

        grouped_results = defaultdict(list)

        # Iterate through features, correlations, and P-values
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

        # Store the top example words and average correlations
        self.category_correlations = {}
        for group, correlations in grouped_results.items():
            avg_correlation = np.mean([c[1] for c in correlations])
            sorted_features = sorted(correlations, key=lambda x: abs(x[1]), reverse=True)
            top_feature = sorted_features[0]  # Select the most correlated feature as an example
            self.category_correlations[group] = {
                "Example Word": top_feature[0],
                "Correlation": avg_correlation,
                "P-Value": top_feature[2]
            }

    def generate_summary_table(self):
        """
        Generate a summary table with categories, example words, correlations, and P-values.
        """
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

        summary_df = pd.DataFrame(summary_data)
        return summary_df

    def visualize_summary_table(self):
        """
        Display the summary table in the notebook.
        """
        summary_table = self.generate_summary_table()
        print(summary_table.to_string(index=False))

# LDA featue analyzer

   def visualize_lda(self, label_filter=None):
        """
        Visualize the LDA model with pyLDAvis, with an option to filter by depressed or non-depressed posts.

        Parameters:
        label_filter (int, optional): Filter by label. Use 1 for depressed, 0 for non-depressed. 
                                       If None, visualize all posts.
        """
        if not self.lda_model or not self.dictionary:
            raise ValueError("LDA model or dictionary not available. Train the model first.")
        
        # Filtered visualization
        if label_filter is not None:
            filtered_docs = [doc for doc, label in zip(self.documents, self.labels) if label == label_filter]
            print(f"Generating LDA visualization for {'depressed' if label_filter == 1 else 'non-depressed'} posts...")

            # Preprocess the filtered documents
            stop_words = set(stopwords.words('english'))
            stemmer = PorterStemmer()
            processed_docs = [
                [
                    stemmer.stem(word) for word in word_tokenize(doc.lower())
                    if word.isalpha() and word not in stop_words
                ]
                for doc in filtered_docs
            ]
            
            # Create a corpus for the filtered documents
            filtered_corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]
            vis = gensimvis.prepare(self.lda_model, filtered_corpus, self.dictionary)
        else:
            # Visualization for all posts
            print("Generating LDA visualization for all posts...")
            vis = gensimvis.prepare(self.lda_model, self.corpus, self.dictionary)

        # Ensure pyLDAvis works in the notebook
        pyLDAvis.enable_notebook()  # Enable visualization in Jupyter Notebook

        # Display visualization
        return pyLDAvis.display(vis)
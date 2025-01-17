import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from empath import Empath
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
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



   

# Empath Feature Extractor
class EmpathFeatureExtractor(FeatureExtractor):
    def __init__(self, documents, labels, output_folder="data/feature_extracted_data"):
        super().__init__(documents, labels, output_folder)
        self.lexicon = Empath()
        self.features = None
        self.correlation_results = None
        self.significant_results = None
        # Categories to focus on based on the origin paper. 
# The categories are divided into linguistic features, psychological processes, personal concerns, and time orientations.
# Since we cannot use the LIWC tool, we will use the Empath tool and define similar categorical features. 

        self.categories = {
            "linguistic_features": [
                "articles", "auxiliary_verbs", "adverbs", "conjunctions", 
                "personal_pronouns", "impersonal_pronouns", "negations", 
                "prepositions", "verbs", "nouns", "adjectives", 
                "comparatives", "superlatives", "modifiers", "function_words", 
                "filler_words", "verb_tense", "slang", "jargon", 
                "formal_language", "casual_language", "exclamations", 
                "contractions", "word_complexity", "sentiment_words"
            ],
                "psychological_processes": {
                "affective": [
                    "positive_emotion", "negative_emotion", "joy", "anger", 
                    "sadness", "anxiety", "fear", "disgust", "love", 
                    "hope", "trust", "excitement", "anticipation", 
                    "relief", "sympathy", "gratitude", "shame", 
                    "guilt", "envy", "pride", "contentment", "confusion",
                    "boredom", "embarrassment", "longing", "nostalgia", 
                    "embarrassment", "frustration", "surprise", "melancholy"
                ],
                "biological": [
                    "body", "health", "illness", "pain", "hygiene", 
                    "fitness", "exercise", "nutrition", "ingestion", 
                    "physical_state", "medicine", "sleep", "sexual", 
                    "aging", "disease", "injury", "hospital", "recovery", 
                    "dieting", "mental_health", "drug_use", "headache", 
                    "fatigue", "hormones", "appetite"
                ],
                "social": [
                    "family", "friends", "relationships", "group_behavior", 
                    "teamwork", "social_media", "communication", "community", 
                    "peer_pressure", "leadership", "parenting", "mentorship", 
                    "marriage", "divorce", "gender_roles", "social_identity", 
                    "cultural_rituals", "networking", "altruism", "conflict", 
                    "social_support", "dominance", "affiliation", "intimacy", 
                    "supportiveness", "competition", "conflict_resolution", 
                    "collaboration", "in-group", "out-group", "prejudice"
                ],
                "cognitive": [
                    "certainty", "doubt", "insight", "cause", "discrepancy", 
                    "problem_solving", "creativity", "self_reflection", "planning", 
                    "memory", "perception", "attention", "reasoning", "thought_process", 
                    "decision_making", "confusion", "learning", "metacognition", "adaptability", 
                    "focus", "perspective", "problem_analysis", "evaluation", "interpretation",
                    "logic", "intelligence", "rational_thought", "intuition", "conceptualization"
                ],
                "drives": [
                    "achievement", "dominance", "affiliation", "control", 
                    "self-esteem", "autonomy", "self-assertion", "power", 
                    "ambition", "conformity", "subordination", "dependence", 
                    "submission", "accomplishment", "independence", "order", 
                    "control_seeking", "status", "prosocial_behavior"
                ],
                "spiritual": [
                    "spirituality", "faith", "beliefs", "sacred", "religion", 
                    "prayer", "meditation", "afterlife", "soul", "divine", 
                    "god", "higher_power", "inspiration", "transcendence", 
                    "morality", "ethics", "rituals", "holiness", "mindfulness"
                ]
            },
            "personal_concerns": [
                "work", "money", "wealth", "shopping", "career", "travel", 
                "home", "school", "education", "violence", "death", 
                "retirement", "spirituality", "family_life", "hobbies", 
                "volunteering", "pets", "entertainment", "parenting", 
                "sports", "adventure", "politics", "environment", 
                "safety", "technology", "materialism", "status", 
                "self_improvement", "learning", "self_growth", "happiness", 
                "life_purpose", "work_life_balance", "stress", "coping", 
                "job_satisfaction", "ambition", "legacy", "job_search", 
                "unemployment", "retirement_plans", "mental_health", "dating", 
                "romantic_relationships", "divorce", "life_stressors", "transitions"
            ],
            "time_orientations": [
                "present", "past", "future", "morning", 
                "afternoon", "evening", "day", "night", 
                "weekdays", "weekends", "seasons", "holidays", 
                "lifespan", "long_term", "short_term", 
                "routine", "historical", "epoch", "momentary", 
                "timeliness", "timelessness", "urgency", 
                "progression", "nostalgia", "anticipation"
            ]
        }

    def extract_empath_features(self):
        features = []
        for doc in self.documents:
            doc_features = {}

            # Linguistic features
            for category in self.categories.get("linguistic_features", []):
                doc_features[category] = self.lexicon.analyze(doc, categories=[category])[category]

            # Psychological processes
            for subcategory, subcategories in self.categories.get("psychological_processes", {}).items():
                for category in subcategories:
                    doc_features[category] = self.lexicon.analyze(doc, categories=[category])[category]

            # Personal concerns
            for category in self.categories.get("personal_concerns", []):
                doc_features[category] = self.lexicon.analyze(doc, categories=[category])[category]

            # Time orientations
            for category in self.categories.get("time_orientations", []):
                doc_features[category] = self.lexicon.analyze(doc, categories=[category])[category]

            features.append(doc_features)

        # Convert to a DataFrame
        self.features = pd.DataFrame(features)

        # Add labels to the features DataFrame
        if len(self.features) == len(self.labels):
            self.features['label'] = self.labels
            print("Added label column to the extracted features.")
        else:
            raise ValueError("Mismatch between the number of features and labels.")

        print(f"Extracted Empath features with shape: {self.features.shape}")

    def analyze_correlation(self):
        if self.features is None:
            raise ValueError("Features must be extracted before analyzing correlations.")

        # Remove constant columns
        constant_columns = self.features.columns[self.features.nunique() == 1]
        self.features.drop(columns=constant_columns, inplace=True, errors='ignore')
        print(f"Removed constant columns: {list(constant_columns)}")

        # Validate labels
        if len(set(self.labels)) == 1:
            raise ValueError("Labels array is constant; cannot compute correlation.")

        correlations, p_values = [], []

        for column in self.features.columns.drop("label"):
            correlation, p_value = pearsonr(self.features[column], self.labels)
            correlations.append(correlation)
            p_values.append(p_value)

        correction_results = multipletests(p_values, alpha=0.05, method="fdr_bh")
        _, corrected_p_values, _, _ = correction_results

        # Create a correlation DataFrame
        self.correlation_results = pd.DataFrame({
            "Feature": self.features.columns.drop("label"),
            "Correlation": correlations,
            "P-Value": p_values,
            "Corrected P-Value": corrected_p_values
        }).sort_values(by="Correlation", key=abs, ascending=False)

    def save_features_and_results(self):
        if self.features is not None:
            feature_file = os.path.join(self.output_folder, "empath_features_with_labels.csv")
            if not os.path.exists(feature_file):
                self.features.to_csv(feature_file, index=False)
                print(f"Saved empath features with labels to {feature_file}.")
            else:
                print(f"Empath features file already exists at {feature_file}.")

        if self.correlation_results is not None:
            correlation_file = os.path.join(self.output_folder, "empath_correlation_results.csv")
            if not os.path.exists(correlation_file):
                self.correlation_results.to_csv(correlation_file, index=False)
                print(f"Saved correlation results to {correlation_file}.")
            else:
                print(f"Correlation results file already exists at {correlation_file}.")



# LDA Feature Extractor
class LDAFeatureExtractor(FeatureExtractor):
    def __init__(self, documents, labels, num_topics=70, passes=15, output_folder="data/feature_extracted_data", random_state=42):
        super().__init__(documents, labels, output_folder)
        self.num_topics = num_topics
        self.passes = passes
        self.random_state = random_state
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
        self.topic_distributions = None
        self.tsne_results = None

    def preprocess_documents(self):
        """
        Preprocess documents: tokenize, remove stopwords, and stem.
        """
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        processed_docs = [
            [
                stemmer.stem(word) for word in word_tokenize(doc.lower())
                if word.isalpha() and word not in stop_words
            ]
            for doc in self.documents
        ]
        return processed_docs

    def filter_docs_by_word_count(self, processed_docs, min_documents=10):
        """
        Filter words that appear in more than 10 documents.
        """
        word_doc_count = defaultdict(int)
        for doc in processed_docs:
            unique_words = set(doc)
            for word in unique_words:
                word_doc_count[word] += 1

        filtered_docs = [
            [word for word in doc if word_doc_count[word] > min_documents]
            for doc in processed_docs
        ]
        return filtered_docs

    def train_lda(self, processed_docs):
        """
        Train the LDA model.
        """
        self.dictionary = corpora.Dictionary(processed_docs)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]
        self.lda_model = LdaModel(
            self.corpus, num_topics=self.num_topics, id2word=self.dictionary, passes=self.passes,
            random_state=self.random_state
        )

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

        # Prepare the LDA features data
        topic_matrix = self.topic_distribution_to_matrix()
        labels_array = np.array(self.labels)
        lda_features_df = pd.DataFrame(topic_matrix)
        lda_features_df['label'] = labels_array

        # Define the filename
        lda_features_file = "lda_topic_distributions_with_labels.csv"

        # Call the base class method for saving
        self.save_to_csv(lda_features_df, lda_features_file)
        
    def run_pipeline(self):
        """
        Complete LDA pipeline: preprocess, train, extract, visualize, and save.
        """
        print("Preprocessing documents...")
        processed_docs = self.preprocess_documents()
        filtered_docs = self.filter_docs_by_word_count(processed_docs)

        print("Training LDA model...")
        self.train_lda(filtered_docs)

        print("Extracting topic distributions...")
        self.extract_topic_distributions()

        print("Saving features...")
        self.save_features()

        print("LDA pipeline complete.")


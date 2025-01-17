
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter

def load_documents_and_labels(folder_paths):
    """
    Load documents and labels from preprocessed posts.
    """
    documents, labels = [], []
    for category, data in folder_paths.items():
        for file_name in os.listdir(data["path"]):
            file_path = os.path.join(data["path"], file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                documents.append(file.read())
                labels.append(data["label"])
    print(f"Loaded {len(documents)} documents.")
    print(f"Labels: {set(labels)}")
    return documents, labels

def define_categories():
    """
    Define the categories used for Empath analysis.
    """
    return {
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

class FeatureAnalyzer:
    def __init__(self, feature_data, labels, output_folder="data/analysis_results"):
        self.feature_data = feature_data
        self.labels = labels
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def compute_frequencies(self, feature_type="unigram"):

        depression_indices = [i for i, label in enumerate(self.labels) if label == 1]
        non_depression_indices = [i for i, label in enumerate(self.labels) if label == 0]

        depression_matrix = self.feature_data[depression_indices]
        non_depression_matrix = self.feature_data[non_depression_indices]

        depression_sums = depression_matrix.sum(axis=0).A1
        non_depression_sums = non_depression_matrix.sum(axis=0).A1

        feature_names = self.feature_data.get_feature_names_out()
        depression_freqs = {feature_names[i]: depression_sums[i] for i in range(len(feature_names))}
        non_depression_freqs = {feature_names[i]: non_depression_sums[i] for i in range(len(feature_names))}

        return depression_freqs, non_depression_freqs

    def get_top_n_features(self, frequencies, top_n=100):

        sorted_features = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:top_n]

    def generate_wordclouds(self):
        depression_unigrams, non_depression_unigrams = self.compute_frequencies(feature_type="unigram")
        depression_bigrams, non_depression_bigrams = self.compute_frequencies(feature_type="bigram")

        wordcloud_data = {
            "Depression - Unigrams": depression_unigrams,
            "Depression - Bigrams": depression_bigrams,
            "Non-Depression - Unigrams": non_depression_unigrams,
            "Non-Depression - Bigrams": non_depression_bigrams
        }

        for title, frequencies in wordcloud_data.items():
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(frequencies)
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(title, fontsize=16)
            plt.show()



class EmpathFeatureAnalyzer:
    def __init__(self):
        self.documents, self.labels = self.load_documents_and_labels()
        self.categories = self.define_categories()
        self.lexicon = Empath()
        self.features = None
        self.correlation_results = None

    def load_documents_and_labels(self):
        folders = {
            "depression": {"path": "data/preprocessed_posts/depression", "label": 1},
            "standard": {"path": "data/preprocessed_posts/standard", "label": 0},
        }
        documents, labels = [], []
        for category, data in folders.items():
            for file_name in os.listdir(data["path"]):
                file_path = os.path.join(data["path"], file_name)
                with open(file_path, 'r', encoding='utf-8') as file:
                    documents.append(file.read())
                    labels.append(data["label"])
        return documents, labels

    def define_categories(self):
        return {
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
            for category in self.categories["linguistic_features"]:
                doc_features[category] = self.lexicon.analyze(doc, categories=[category])[category]
            for subcategory, subfeatures in self.categories["psychological_processes"].items():
                for feature in subfeatures:
                    doc_features[feature] = self.lexicon.analyze(doc, categories=[feature])[feature]
            for category in self.categories["personal_concerns"]:
                doc_features[category] = self.lexicon.analyze(doc, categories=[category])[category]
            for category in self.categories["time_orientations"]:
                doc_features[category] = self.lexicon.analyze(doc, categories=[category])[category]
            features.append(doc_features)
        self.features = pd.DataFrame(features)

    def analyze_correlation(self):
        correlations = []
        for column in self.features.columns:
            corr = np.corrcoef(self.features[column], self.labels)[0, 1]
            correlations.append(corr)
        self.correlation_results = pd.DataFrame({
            "Feature": self.features.columns,
            "Correlation": correlations
        }).sort_values(by="Correlation", ascending=False)


class NGramFeatureAnalyzer:
    def __init__(self):
        self.documents, self.labels = self.load_documents_and_labels()
        self.unigram_vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
        self.bigram_vectorizer = TfidfVectorizer(ngram_range=(2, 2), stop_words='english')

    def load_documents_and_labels(self):
        folders = {
            "depression": {"path": "data/preprocessed_posts/depression", "label": 1},
            "standard": {"path": "data/preprocessed_posts/standard", "label": 0},
        }
        documents, labels = [], []
        for category, data in folders.items():
            for file_name in os.listdir(data["path"]):
                file_path = os.path.join(data["path"], file_name)
                with open(file_path, 'r', encoding='utf-8') as file:
                    documents.append(file.read())
                    labels.append(data["label"])
        return documents, labels

    def extract_ngrams(self):
        self.unigram_matrix = self.unigram_vectorizer.fit_transform(self.documents)
        self.bigram_matrix = self.bigram_vectorizer.fit_transform(self.documents)
        self.unigram_features = self.unigram_vectorizer.get_feature_names_out()
        self.bigram_features = self.bigram_vectorizer.get_feature_names_out()

    def generate_wordclouds(self):
        for name, matrix, features in [("Unigrams", self.unigram_matrix, self.unigram_features),
                                        ("Bigrams", self.bigram_matrix, self.bigram_features)]:
            freqs = np.array(matrix.sum(axis=0)).flatten()
            freq_dict = {features[i]: freqs[i] for i in range(len(features))}
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(freq_dict)
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f"{name} Word Cloud", fontsize=16)
            plt.show()


class LDAFeatureAnalyzer:
    def __init__(self, lda_model, topic_matrix):
        self.documents, self.labels = self.load_documents_and_labels()
        self.lda_model = lda_model
        self.topic_matrix = topic_matrix

    def load_documents_and_labels(self):
        folders = {
            "depression": {"path": "data/preprocessed_posts/depression", "label": 1},
            "standard": {"path": "data/preprocessed_posts/standard", "label": 0},
        }
        documents, labels = [], []
        for category, data in folders.items():
            for file_name in os.listdir(data["path"]):
                file_path = os.path.join(data["path"], file_name)
                with open(file_path, 'r', encoding='utf-8') as file:
                    documents.append(file.read())
                    labels.append(data["label"])
        return documents, labels

    def run_tsne(self):
        tsne = TSNE(n_components=2, random_state=42)
        return tsne.fit_transform(self.topic_matrix)

    def cluster_topics(self):
        kmeans = KMeans(n_clusters=5, random_state=42)
        return kmeans.fit_predict(self.topic_matrix)
    
    import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import defaultdict
from empath import Empath

class EmpathFeatureAnalyzer:
    def __init__(self):
        self.documents, self.labels = self.load_documents_and_labels()
        self.categories = self.define_categories()
        self.lexicon = Empath()
        self.features = None
        self.correlation_results = None

    def load_documents_and_labels(self):
        folders = {
            "depression": {"path": "data/preprocessed_posts/depression", "label": 1},
            "standard": {"path": "data/preprocessed_posts/standard", "label": 0},
        }
        documents, labels = [], []
        for category, data in folders.items():
            for file_name in os.listdir(data["path"]):
                file_path = os.path.join(data["path"], file_name)
                with open(file_path, 'r', encoding='utf-8') as file:
                    documents.append(file.read())
                    labels.append(data["label"])
        print(f"Loaded {len(documents)} documents.")
        return documents, labels

    def define_categories(self):
        return {
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
                    "frustration", "surprise", "melancholy"
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
            for category in self.categories["linguistic_features"]:
                doc_features[category] = self.lexicon.analyze(doc, categories=[category])[category]
            for subcategory, subfeatures in self.categories["psychological_processes"].items():
                for feature in subfeatures:
                    doc_features[feature] = self.lexicon.analyze(doc, categories=[feature])[feature]
            for category in self.categories["personal_concerns"]:
                doc_features[category] = self.lexicon.analyze(doc, categories=[category])[category]
            for category in self.categories["time_orientations"]:
                doc_features[category] = self.lexicon.analyze(doc, categories=[category])[category]
            features.append(doc_features)
        self.features = pd.DataFrame(features)

    def analyze_correlation(self):
        correlations = []
        for column in self.features.columns:
            corr = np.corrcoef(self.features[column], self.labels)[0, 1]
            correlations.append(corr)
        self.correlation_results = pd.DataFrame({
            "Feature": self.features.columns,
            "Correlation": correlations
        }).sort_values(by="Correlation", ascending=False)

class NGramFeatureAnalyzer:
    def __init__(self):
        self.documents, self.labels = self.load_documents_and_labels()
        self.unigram_vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
        self.bigram_vectorizer = TfidfVectorizer(ngram_range=(2, 2), stop_words='english')

    def load_documents_and_labels(self):
        folders = {
            "depression": {"path": "data/preprocessed_posts/depression", "label": 1},
            "standard": {"path": "data/preprocessed_posts/standard", "label": 0},
        }
        documents, labels = [], []
        for category, data in folders.items():
            for file_name in os.listdir(data["path"]):
                file_path = os.path.join(data["path"], file_name)
                with open(file_path, 'r', encoding='utf-8') as file:
                    documents.append(file.read())
                    labels.append(data["label"])
        return documents, labels

    def extract_ngrams(self):
        self.unigram_matrix = self.unigram_vectorizer.fit_transform(self.documents)
        self.bigram_matrix = self.bigram_vectorizer.fit_transform(self.documents)
        self.unigram_features = self.unigram_vectorizer.get_feature_names_out()
        self.bigram_features = self.bigram_vectorizer.get_feature_names_out()

    def generate_wordclouds(self):
        for name, matrix, features in [("Unigrams", self.unigram_matrix, self.unigram_features),
                                        ("Bigrams", self.bigram_matrix, self.bigram_features)]:
            freqs = np.array(matrix.sum(axis=0)).flatten()
            freq_dict = {features[i]: freqs[i] for i in range(len(features))}
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(freq_dict)
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f"{name} Word Cloud", fontsize=16)
            plt.show()

class LDAFeatureAnalyzer:
    def __init__(self, lda_model, topic_matrix):
        self.documents, self.labels = self.load_documents_and_labels()
        self.lda_model = lda_model
        self.topic_matrix = topic_matrix

    def load_documents_and_labels(self):
        folders = {
            "depression": {"path": "data/preprocessed_posts/depression", "label": 1},
            "standard": {"path": "data/preprocessed_posts/standard", "label": 0},
        }
        documents, labels = [], []
        for category, data in folders.items():
            for file_name in os.listdir(data["path"]):
                file_path = os.path.join(data["path"], file_name)
                with open(file_path, 'r', encoding='utf-8') as file:
                    documents.append(file.read())
                    labels.append(data["label"])
        return documents, labels

    def run_tsne(self):
        tsne = TSNE(n_components=2, random_state=42)
        return tsne.fit_transform(self.topic_matrix)

    def cluster_topics(self):
        kmeans = KMeans(n_clusters=5, random_state=42)
        return kmeans.fit_predict(self.topic_matrix)
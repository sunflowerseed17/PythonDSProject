###############################################################################
#  IMPORTS
###############################################################################

import os
import pandas as pd
from empath import Empath
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
from feature_extraction.base_feature_extraction_func import FeatureExtractor

###############################################################################
# EMPATH FEATURE-EXTRACTOR
###############################################################################

class EmpathFeatureExtractor(FeatureExtractor):
    def __init__(self, *, output_folder="data/feature_extracted_data", folders=None):
        super().__init__(folders=folders, output_folder=output_folder)
        self.lexicon = Empath()
        self.features = None
        self.correlation_results = None
        self.significant_results = None

        # Categories to focus on 
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

    def generate_correlation_table(self):
        if self.correlation_results is None:
            raise ValueError("Correlation results must be calculated before generating the table.")

        results = []

        for _, row in self.correlation_results.iterrows():
            feature = row["Feature"]
            correlation = row["Correlation"]
            p_value = row["P-Value"]

            # Look through each main category or subcategory
            for category, features in self.categories.items():
                if isinstance(features, dict):  
                    for subcategory, subfeatures in features.items():
                        if feature in subfeatures:
                            results.append((f"{category} - {subcategory}", feature, correlation, p_value))
                elif feature in features:
                    results.append((category, feature, correlation, p_value))

        correlation_table = pd.DataFrame(results, columns=["Empath Category", "Example Word", "Correlation", "P-Value"])
        return correlation_table
    
    def save_correlation_table(self, output_folder):
        
        correlation_table = self.generate_correlation_table()
        file_path = os.path.join(output_folder, "Empath_Correlation_Table.csv")
        correlation_table.to_csv(file_path, index=False)
        print(f"Saved Empath correlation table: {file_path}")

    def save_features_and_results(self, overwrite=False):
        if self.features is not None:
            feature_file = os.path.join(self.output_folder, "empath_features_with_labels.csv")
            if overwrite or not os.path.exists(feature_file):
                self.features.to_csv(feature_file, index=False)
                print(f"Saved empath features with labels to {feature_file}.")
            else:
                print(f"Empath features file already exists at {feature_file}.")
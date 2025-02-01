import logging
from model_training_func import ModelTrainer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

###############################################################################
# LOGGER CONFIGURATION
###############################################################################
logging.basicConfig(
    level=logging.INFO,  # Adjust as needed
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

###############################################################################
# CONFIGURATION CONSTANTS
###############################################################################
FEATURE_COMBINATIONS = {
    "Empath": ["data/feature_extracted_data/empath_features_with_labels.csv"],
    "LDA": ["data/feature_extracted_data/lda_topic_distributions_with_labels.csv"],
    "Unigram": ["data/feature_extracted_data/unigram_features_with_labels.csv"],
    "Bigram": ["data/feature_extracted_data/bigram_features_with_labels.csv"],
    "Empath+LDA+Unigram": [
        "data/feature_extracted_data/empath_features_with_labels.csv",
        "data/feature_extracted_data/lda_topic_distributions_with_labels.csv",
        "data/feature_extracted_data/unigram_features_with_labels.csv"
    ],
    "Empath+LDA+Bigram": [
        "data/feature_extracted_data/empath_features_with_labels.csv",
        "data/feature_extracted_data/lda_topic_distributions_with_labels.csv",
        "data/feature_extracted_data/bigram_features_with_labels.csv"
    ]
}

MODELS_CONFIG = {
    "LR": {"model_class": LogisticRegression, "params": {'max_iter': 500, 'random_state': 42}},
    "SVM": {"model_class": SVC, "params": {'C': 1.0, 'kernel': 'linear', 'random_state': 42}},
    "RF": {"model_class": RandomForestClassifier, "params": {'n_estimators': 100, 'random_state': 42}},
    "AdaBoost": {"model_class": AdaBoostClassifier, "params": {'n_estimators': 50, 'random_state': 42}},
    "MLP": {"model_class": MLPClassifier, "params": {'hidden_layer_sizes': (50,), 'activation': 'relu', 'solver': 'adam', 'random_state': 42, 'max_iter': 1000, 'early_stopping': True}}
}

OUTPUT_FILENAME = "model_results.png"

###############################################################################
# MAIN FUNCTION
###############################################################################
def main():
    logger.info("Starting model training and evaluation...")
    ModelTrainer.run_and_save_results(FEATURE_COMBINATIONS, MODELS_CONFIG, OUTPUT_FILENAME)
    logger.info("Model training complete. Results saved as '%s'.", OUTPUT_FILENAME)

if __name__ == "__main__":
    main()
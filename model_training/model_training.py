from model_training_func import ModelTrainer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Define file paths for feature sets
feature_combinations = {
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

# Define model configurations
models_config = {
    "LR": {"model_class": LogisticRegression, "params": {'max_iter': 500, 'random_state': 42}},
    "SVM": {"model_class": SVC, "params": {'C': 1.0, 'kernel': 'linear', 'random_state': 42}},
    "RF": {"model_class": RandomForestClassifier, "params": {'n_estimators': 100, 'random_state': 42}},
    "AdaBoost": {"model_class": AdaBoostClassifier, "params": {'n_estimators': 50, 'random_state': 42}},
    "MLP": {"model_class": MLPClassifier, "params": {'hidden_layer_sizes': (50,), 'activation': 'relu', 'solver': 'adam', 'random_state': 42, 'max_iter': 1000, 'early_stopping': True}}   
}

# Output file for the results table
output_filename = "model_results.png"

# Run the training and save results
ModelTrainer.run_and_save_results(feature_combinations, models_config, output_filename)
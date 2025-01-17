import os
from feature_extraction_func import NGramFeatureExtractor, EmpathFeatureExtractor, LDAFeatureExtractor

# Loading data
if __name__ == "__main__":
    # Load documents and labels
    folders = {
        "depression": {"path": "data/preprocessed_posts/depression", "label": 1},
        "standard": {"path": "data/preprocessed_posts/standard", "label": 0},
        "breastcancer": {"path": "data/preprocessed_posts/breastcancer", "label": 2},
    }

    documents, labels = [], []
    for category, data in folders.items():
        for file_name in os.listdir(data["path"]):
            file_path = os.path.join(data["path"], file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                documents.append(file.read())
                labels.append(data["label"])
    print(f"Loaded {len(documents)} documents.")
    print(f"Labels: {set(labels)}")

    # Initialize the N-Gram Extractor
    ngram_extractor = NGramFeatureExtractor(documents, labels)
    ngram_extractor.extract_features()
    ngram_extractor.save_features()

    # Initialize Empath Feature Extractor
    empath_extractor = EmpathFeatureExtractor(documents, labels)
    empath_extractor.extract_empath_features()
    empath_extractor.analyze_correlation()
    empath_extractor.save_features_and_results()

    # Initialize LDA Feature Extractor
    lda_extractor = LDAFeatureExtractor(documents, labels)
    lda_extractor.run_pipeline()
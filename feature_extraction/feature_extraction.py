from ngram_feature_extraction_func import NGramFeatureExtractor
from empath_feature_extraction_func import EmpathFeatureExtractor
from lda_feature_extraction_func import LDAFeatureExtractor

folders = {
            "depression": {"path": "data/preprocessed_posts/depression", "label": 1},
            "standard": {"path": "data/preprocessed_posts/standard", "label": 0},
            "breastcancer": {"path": "data/preprocessed_posts/breastcancer", "label": 2},
}

# Initialize and run N-Gram Feature Extraction
ngram_extractor = NGramFeatureExtractor(folders=folders, output_folder="data/feature_extracted_data")
ngram_extractor.extract_features()
ngram_extractor.save_features()

# Initialize and run Empath Feature Extraction
empath_extractor = EmpathFeatureExtractor(folders=folders, output_folder="data/feature_extracted_data")
empath_extractor.extract_empath_features()
empath_extractor.analyze_correlation()
empath_extractor.save_features_and_results()

# Initialize and run LDA Feature Extraction
lda_extractor = LDAFeatureExtractor(folders=folders, output_folder="data/feature_extracted_data")
lda_extractor.run_feature_extraction()

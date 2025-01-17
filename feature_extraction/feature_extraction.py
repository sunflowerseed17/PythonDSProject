from feature_extraction_func import NGramFeatureExtractor, EmpathFeatureExtractor, LDAFeatureExtractor

# Initialize the N-Gram Extractor
ngram_extractor = NGramFeatureExtractor()
ngram_extractor.extract_features()
ngram_extractor.save_features()

# Initialize Empath Feature Extractor
empath_extractor = EmpathFeatureExtractor()
empath_extractor.extract_empath_features()
empath_extractor.analyze_correlation()
empath_extractor.save_features_and_results()

# Initialize LDA Feature Extractor
lda_extractor = LDAFeatureExtractor()
lda_extractor.run_pipeline()
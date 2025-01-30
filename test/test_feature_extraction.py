import unittest
from feature_extraction.ngram_feature_extraction_func import NGramFeatureExtractor
from feature_extraction.empath_feature_extraction_func import EmpathFeatureExtractor
from feature_extraction.lda_feature_extraction_func import LDAFeatureExtractor

class TestFeatureExtraction(unittest.TestCase):

    def setUp(self):
        """Set up feature extraction paths."""
        self.folders = {
            "depression": {"path": "test/test_data/preprocessed/depression", "label": 1},
            "standard": {"path": "test/test_data/preprocessed/standard", "label": 0},
            "breastcancer": {"path": "test/test_data/preprocessed/breastcancer", "label": 2},
        }

    def test_ngram_extraction(self):
        """Test that n-gram feature extraction runs without errors."""
        extractor = NGramFeatureExtractor(folders=self.folders)
        unigram_matrix, bigram_matrix, combined_matrix = extractor.extract_features()
        self.assertIsNotNone(unigram_matrix, "Unigram matrix is None")
        self.assertIsNotNone(bigram_matrix, "Bigram matrix is None")
        self.assertIsNotNone(combined_matrix, "Combined matrix is None")

    def test_empath_extraction(self):
        """Test that Empath feature extraction runs without errors."""
        extractor = EmpathFeatureExtractor(folders=self.folders)
        extractor.extract_empath_features()
        self.assertIsNotNone(extractor.features, "Empath features are None")

    def test_lda_extraction(self):
        """Test that LDA feature extraction runs without errors."""
        extractor = LDAFeatureExtractor(folders=self.folders, num_topics=5)
        extractor.run_feature_extraction()
        self.assertIsNotNone(extractor.lda_model, "LDA model is None")

if __name__ == "__main__":
    unittest.main()

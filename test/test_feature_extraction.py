import unittest
import logging
from feature_extraction.ngram_feature_extraction_func import NGramFeatureExtractor
from feature_extraction.empath_feature_extraction_func import EmpathFeatureExtractor
from feature_extraction.lda_feature_extraction_func import LDAFeatureExtractor

import nltk
nltk.download('punkt_tab', quiet=True)

###############################################################################
# LOGGER CONFIGURATION
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class TestFeatureExtraction(unittest.TestCase):

    def setUp(self):
        """Set up feature extraction paths for testing."""
        self.folders = {
            "depression": {"path": "test/test_data/preprocessed/depression", "label": 1},
            "standard": {"path": "test/test_data/preprocessed/standard", "label": 0},
            "breastcancer": {"path": "test/test_data/preprocessed/breastcancer", "label": 2},
        }
        logger.info("Test setup complete with folders: %s", self.folders)

    def test_ngram_extraction(self):
        """Test that n-gram feature extraction runs without errors."""
        logger.info("Starting n-gram extraction test.")
        extractor = NGramFeatureExtractor(folders=self.folders)
        unigram_matrix, bigram_matrix, combined_matrix = extractor.extract_features()
        self.assertIsNotNone(unigram_matrix, "Unigram matrix is None")
        self.assertIsNotNone(bigram_matrix, "Bigram matrix is None")
        self.assertIsNotNone(combined_matrix, "Combined matrix is None")
        logger.info("N-gram extraction test passed.")

    def test_empath_extraction(self):
        """Test that Empath feature extraction runs without errors."""
        logger.info("Starting Empath extraction test.")
        extractor = EmpathFeatureExtractor(folders=self.folders)
        extractor.extract_empath_features()
        self.assertIsNotNone(extractor.features, "Empath features are None")
        logger.info("Empath extraction test passed.")

    def test_lda_extraction(self):
        """Test that LDA feature extraction runs without errors."""
        logger.info("Starting LDA extraction test.")
        extractor = LDAFeatureExtractor(folders=self.folders, num_topics=5)
        extractor.run_feature_extraction()
        self.assertIsNotNone(extractor.lda_model, "LDA model is None")
        logger.info("LDA extraction test passed.")

if __name__ == "__main__":
    unittest.main()
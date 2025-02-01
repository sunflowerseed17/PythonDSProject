import sys
import os
import unittest
import logging

# Ensure the project root is in the sys.path so that the feature_extraction package is found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# Import the feature analysis modules using the proper package paths.
from feature_extraction.ngram_feature_extraction_func import NGramFeatureExtractor
from feature_extraction.empath_feature_extraction_func import EmpathFeatureExtractor
from feature_extraction.lda_feature_extraction_func import LDAFeatureExtractor
from feature_extraction.base_feature_extraction_func import generate_summary_table, generate_empath_table

###############################################################################
# LOGGER CONFIGURATION
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class TestFeatureAnalysis(unittest.TestCase):
    def setUp(self):
        """Set up a temporary directory for output files for analysis tests."""
        self.temp_dir = os.path.abspath("test/test_data/temp_analysis")
        os.makedirs(self.temp_dir, exist_ok=True)
        logger.info("Created temporary directory for analysis tests: %s", self.temp_dir)

    def tearDown(self):
        """Clean up the temporary directory after tests."""
        # Optionally remove the temporary directory:
        # import shutil
        # shutil.rmtree(self.temp_dir)
        logger.info("Tear down of analysis tests complete.")

    def test_ngram_analysis(self):
        """Test that the N-Gram analysis functions run without error and produce outputs."""
        # Create dummy data for testing
        dummy_docs = [
            "This is a test sentence for n-gram analysis.",
            "Another test sentence with extra text data."
        ]
        extractor = NGramFeatureExtractor()
        extractor.documents = dummy_docs
        extractor.labels = [0, 1]

        # Run extraction and save word clouds to the temporary directory
        extractor.extract_features()
        extractor.save_wordclouds(self.temp_dir)

        # Verify that at least one PNG file exists in the temp directory
        png_files = [f for f in os.listdir(self.temp_dir) if f.endswith(".png")]
        self.assertGreater(len(png_files), 0, "No word cloud images were produced by N-Gram analysis.")
        logger.info("N-Gram analysis test passed.")

    def test_empath_analysis(self):
        """Test that the Empath analysis functions run without error and produce outputs."""
        dummy_docs = [
            "This is a test sentence for empath analysis with emotion.",
            "Another test sentence with different emotional tone."
        ]
        extractor = EmpathFeatureExtractor()
        extractor.documents = dummy_docs
        extractor.labels = [0, 1]

        extractor.extract_empath_features()
        extractor.analyze_correlation()
        extractor.save_correlation_table(self.temp_dir)

        # Verify that the correlation CSV file exists
        corr_csv = os.path.join(self.temp_dir, "Empath_Correlation_Table.csv")
        self.assertTrue(os.path.exists(corr_csv), "Empath correlation table CSV was not created.")
        logger.info("Empath analysis test passed.")

    def test_lda_analysis(self):
        """Test that the LDA analysis functions run without error and produce outputs."""
        dummy_docs = [
            "This is a test sentence for LDA analysis.",
            "LDA test sentence regarding topic modeling.",
            "Additional document text for topic extraction."
        ]
        # Assume that label '1' indicates depressed posts (for analysis purposes)
        extractor = LDAFeatureExtractor(num_topics=3, passes=1)
        extractor.documents = dummy_docs
        extractor.labels = [1, 0, 1]

        try:
            extractor.run_feature_analysis()
        except Exception as e:
            self.fail(f"LDA analysis pipeline failed: {e}")

        # Generate a topic table image and verify it exists
        topic_table_path = os.path.join(self.temp_dir, "topic_table_depressed.png")
        extractor.generate_topic_table(output_file=topic_table_path)
        self.assertTrue(os.path.exists(topic_table_path), "LDA topic table image was not created.")
        logger.info("LDA analysis test passed.")

    def test_summary_and_empath_tables(self):
        """Test that summary and Empath tables are generated successfully."""
        # Set up dummy data for all extractors.
        ngram_extractor = NGramFeatureExtractor()
        ngram_extractor.documents = [
            "This is a test sentence for n-gram analysis.",
            "Another sentence for testing."
        ]
        ngram_extractor.labels = [0, 1]
        ngram_extractor.extract_features()

        empath_extractor = EmpathFeatureExtractor()
        empath_extractor.documents = [
            "Test sentence for empath analysis.",
            "Another empath test sentence."
        ]
        empath_extractor.labels = [0, 1]
        empath_extractor.extract_empath_features()
        empath_extractor.analyze_correlation()

        lda_extractor = LDAFeatureExtractor(num_topics=3, passes=1)
        lda_extractor.documents = [
            "Test sentence for LDA analysis.",
            "LDA analysis test sentence.",
            "Extra document text for topic modeling."
        ]
        lda_extractor.labels = [1, 0, 1]
        try:
            lda_extractor.run_feature_analysis()
        except Exception as e:
            self.fail(f"LDA analysis in summary table generation failed: {e}")

        # Generate summary table and verify it exists.
        summary_table_path = os.path.join(self.temp_dir, "summary_table.png")
        generate_summary_table(ngram_extractor, empath_extractor, lda_extractor, summary_table_path)
        self.assertTrue(os.path.exists(summary_table_path), "Summary table image was not created.")

        # Generate an Empath table and verify it exists.
        empath_table_path = os.path.join(self.temp_dir, "empath_table.png")
        corr_csv = os.path.join(self.temp_dir, "Empath_Correlation_Table.csv")
        generate_empath_table(corr_csv, empath_table_path)
        self.assertTrue(os.path.exists(empath_table_path), "Empath table image was not created.")
        logger.info("Summary and Empath table tests passed.")

if __name__ == "__main__":
    unittest.main()
import os
import logging

from ngram_feature_extraction_func import NGramFeatureExtractor
from empath_feature_extraction_func import EmpathFeatureExtractor
from lda_feature_extraction_func import LDAFeatureExtractor

###############################################################################
# CONFIGURATION CONSTANTS
###############################################################################
PREPROCESSED_POSTS_DIR = os.path.join("data", "preprocessed_posts")
FEATURE_EXTRACTED_DATA_DIR = os.path.join("data", "feature_extracted_data")

FOLDERS = {
    "depression": {
        "path": os.path.join(PREPROCESSED_POSTS_DIR, "depression"),
        "label": 1
    },
    "standard": {
        "path": os.path.join(PREPROCESSED_POSTS_DIR, "standard"),
        "label": 0
    },
    "breastcancer": {
        "path": os.path.join(PREPROCESSED_POSTS_DIR, "breastcancer"),
        "label": 2
    },
}

###############################################################################
# LOGGER CONFIGURATION
###############################################################################
logging.basicConfig(
    level=logging.INFO,  # Adjust the log level as needed
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

###############################################################################
# FEATURE EXTRACTION PIPELINE
###############################################################################
def run_feature_extraction():
    # N-Gram Feature Extraction
    logger.info("Starting N-Gram Feature Extraction...")
    ngram_extractor = NGramFeatureExtractor(folders=FOLDERS, output_folder=FEATURE_EXTRACTED_DATA_DIR)
    ngram_extractor.extract_features()
    ngram_extractor.save_features()
    logger.info("N-Gram Feature Extraction completed.")

    # Empath Feature Extraction
    logger.info("Starting Empath Feature Extraction...")
    empath_extractor = EmpathFeatureExtractor(folders=FOLDERS, output_folder=FEATURE_EXTRACTED_DATA_DIR)
    empath_extractor.extract_empath_features()
    empath_extractor.analyze_correlation()
    empath_extractor.save_features_and_results()
    logger.info("Empath Feature Extraction completed.")

    # LDA Feature Extraction
    logger.info("Starting LDA Feature Extraction...")
    lda_extractor = LDAFeatureExtractor(folders=FOLDERS, output_folder=FEATURE_EXTRACTED_DATA_DIR)
    lda_extractor.run_feature_extraction()
    logger.info("LDA Feature Extraction completed.")

if __name__ == "__main__":
    run_feature_extraction()
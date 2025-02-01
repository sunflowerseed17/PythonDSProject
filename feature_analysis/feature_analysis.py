###############################################################################
#  IMPORTS
###############################################################################
import sys
import os
import logging

# Append the feature extraction directory to the system path.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../feature_extraction"))

from ngram_feature_extraction_func import NGramFeatureExtractor
from empath_feature_extraction_func import EmpathFeatureExtractor
from lda_feature_extraction_func import LDAFeatureExtractor
from base_feature_extraction_func import generate_empath_table, generate_summary_table

###############################################################################
# CONFIGURATION CONSTANTS
###############################################################################
# Output folder for analysis results
OUTPUT_FOLDER = "outputs"

# Filenames for generated summary and tables
SUMMARY_TABLE_FILENAME = "summary_table.png"
EMPATH_CORR_TABLE_CSV = "Empath_Correlation_Table.csv"
EMPATH_TABLE_IMG = "empath_table.png"

# LDA configuration
NUM_TOPICS_LDA = 20

###############################################################################
# LOGGER CONFIGURATION
###############################################################################
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more verbosity if needed.
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

###############################################################################
# OUTPUT SETUP
###############################################################################
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

###############################################################################
# ANALYSIS PIPELINE
###############################################################################

# N-Gram Analysis
logger.info("Starting N-Gram Analysis...")
ngram_extractor = NGramFeatureExtractor()
ngram_extractor.extract_features()
ngram_extractor.save_wordclouds(OUTPUT_FOLDER)
logger.info("N-Gram Analysis complete.")

# Empath Analysis
logger.info("Starting Empath Analysis...")
empath_extractor = EmpathFeatureExtractor()
empath_extractor.extract_empath_features()
empath_extractor.analyze_correlation()
empath_extractor.save_correlation_table(OUTPUT_FOLDER)
logger.info("Empath Analysis complete.")

# LDA Analysis with Enhanced t-SNE Visualization
logger.info("Starting LDA Analysis...")
lda_extractor = LDAFeatureExtractor(num_topics=NUM_TOPICS_LDA)
lda_extractor.run_feature_analysis()
logger.info("LDA Analysis complete.")

# Generate Summary Table for All Features
logger.info("Generating summary table for feature extraction methods...")
summary_table_output = os.path.join(OUTPUT_FOLDER, SUMMARY_TABLE_FILENAME)
generate_summary_table(
    ngram_extractor, empath_extractor, lda_extractor, summary_table_output
)
logger.info("Summary table saved.")

# Generate Empath Correlation Table
logger.info("Generating Empath correlation table...")
empath_corr_csv = os.path.join(OUTPUT_FOLDER, EMPATH_CORR_TABLE_CSV)
empath_table_img = os.path.join(OUTPUT_FOLDER, EMPATH_TABLE_IMG)
generate_empath_table(empath_corr_csv, empath_table_img)
logger.info("Empath correlation table saved.")

###############################################################################
# COMPLETION MESSAGE
###############################################################################
logger.info("All feature extraction and analysis steps are complete.")
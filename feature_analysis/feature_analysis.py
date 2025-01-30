###############################################################################
#  IMPORTS
###############################################################################
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../feature_extraction"))
from ngram_feature_extraction_func import NGramFeatureExtractor
from empath_feature_extraction_func import EmpathFeatureExtractor
from lda_feature_extraction_func import LDAFeatureExtractor
from base_feature_extraction_func import generate_empath_table, generate_summary_table

###############################################################################
#  OUTPUT SETUP
###############################################################################
output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)

###############################################################################
#  ANALYSIS PIPELINE
###############################################################################

# N-Gram Analysis
print("Starting N-Gram Analysis...")
ngram_extractor = NGramFeatureExtractor()
ngram_extractor.extract_features()
ngram_extractor.save_wordclouds(output_folder)
print("N-Gram Analysis complete.\n")

# Empath Analysis
print("Starting Empath Analysis...")
empath_extractor = EmpathFeatureExtractor()
empath_extractor.extract_empath_features()
empath_extractor.analyze_correlation()
empath_extractor.save_correlation_table(output_folder)
print("Empath Analysis complete.\n")

# LDA Analysis with Enhanced t-SNE Visualization
print("Starting LDA Analysis...")
lda_extractor = LDAFeatureExtractor(num_topics=20)
lda_extractor.run_feature_analysis()
print("LDA Analysis complete.\n")

# Generate Summary Table for All Features
print("Generating summary table for feature extraction methods...")
generate_summary_table(
    ngram_extractor, empath_extractor, lda_extractor, f"{output_folder}/summary_table.png"
)
print("Summary table saved.\n")

# Generate Empath Correlation Table
print("Generating Empath correlation table...")
generate_empath_table(
    f"{output_folder}/Empath_Correlation_Table.csv", f"{output_folder}/empath_table.png"
)
print("Empath correlation table saved.\n")

###############################################################################
#  COMPLETION MESSAGE
###############################################################################
print("All feature extraction and analysis steps are complete.")
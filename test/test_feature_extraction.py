import pytest
import os
import shutil
from feature_extraction_func import NGramFeatureExtractor, EmpathFeatureExtractor, LDAFeatureExtractor


@pytest.fixture
def setup_test_environment(tmp_path):

    depression_dir = tmp_path / "test/data/preprocessed_posts/depression"
    standard_dir = tmp_path / "test/data/preprocessed_posts/standard"
    breastcancer_dir = tmp_path / "test/data/preprocessed_posts/breastcancer"

    # Create directories
    depression_dir.mkdir(parents=True, exist_ok=True)
    standard_dir.mkdir(parents=True, exist_ok=True)
    breastcancer_dir.mkdir(parents=True, exist_ok=True)

    # Write some mock text files
    (depression_dir / "post1.txt").write_text("I feel so sad and alone.")
    (standard_dir / "post2.txt").write_text("It’s a sunny day. I went to the park.")
    (breastcancer_dir / "post3.txt").write_text("Breast cancer treatment options are expanding.")

    return tmp_path


def test_ngram_feature_extractor(setup_test_environment):
    # Initialize the NGramFeatureExtractor
    output_folder = setup_test_environment / "test/data/feature_extracted_data"
    extractor = NGramFeatureExtractor(output_folder=str(output_folder))

    # Test feature extraction
    unigram_matrix, bigram_matrix, combined_matrix = extractor.extract_features()

    # Assertions
    assert unigram_matrix.shape[0] == 3  # 3 documents
    assert bigram_matrix.shape[0] == 3
    assert combined_matrix.shape[0] == 3

    # Test saving features
    extractor.save_features()
    assert (output_folder / "unigram_features_with_labels.csv").exists()
    assert (output_folder / "bigram_features_with_labels.csv").exists()


def test_empath_feature_extractor(setup_test_environment):
    # Initialize the EmpathFeatureExtractor
    output_folder = setup_test_environment / "test/data/feature_extracted_data"
    extractor = EmpathFeatureExtractor(output_folder=str(output_folder))

    # Test feature extraction
    extractor.extract_empath_features()

    # Assertions
    assert extractor.features is not None
    assert "label" in extractor.features.columns

    # Test saving features and correlation results
    extractor.save_features_and_results(overwrite=True)
    assert (output_folder / "empath_features_with_labels.csv").exists()


def test_lda_feature_extractor(setup_test_environment):
    # Initialize the LDAFeatureExtractor
    output_folder = setup_test_environment / "test/data/feature_extracted_data"
    extractor = LDAFeatureExtractor(num_topics=2, output_folder=str(output_folder))

    # Run the feature extraction pipeline
    extractor.run_feature_extraction_pipeline()

    # Assertions
    assert extractor.lda_model is not None
    assert (output_folder / "lda_topic_distributions_with_labels.csv").exists()

    # Run the feature analysis pipeline
    extractor.run_feature_analysis_pipeline()

    # Assertions for analysis outputs
    assert (output_folder / "topic_table.png").exists()
    assert (output_folder / "tsne_visualization.png").exists()
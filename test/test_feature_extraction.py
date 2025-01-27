import pytest
import pandas as pd
from feature_extraction.feature_extraction_func import NGramFeatureExtractor, EmpathFeatureExtractor, LDAFeatureExtractor


@pytest.fixture
def setup_test_environment(tmp_path):
    # Mock directories
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
    output_folder = setup_test_environment / "test/data/feature_extracted_data"
    extractor = NGramFeatureExtractor(output_folder=str(output_folder))

    # Test feature extraction
    unigram_matrix, bigram_matrix, combined_matrix = extractor.extract_features()

    # Assertions on matrix dimensions
    assert unigram_matrix.shape[0] == 3  # 3 documents
    assert unigram_matrix.shape[1] > 0  # Ensure features are extracted
    assert bigram_matrix.shape[0] == 3
    assert combined_matrix.shape[0] == 3

    # Test saving features
    extractor.save_features()
    unigram_path = output_folder / "unigram_features_with_labels.csv"
    bigram_path = output_folder / "bigram_features_with_labels.csv"
    assert unigram_path.exists()
    assert bigram_path.exists()

    # Validate saved CSV contents
    unigram_df = pd.read_csv(unigram_path)
    bigram_df = pd.read_csv(bigram_path)
    assert "label" in unigram_df.columns
    assert "label" in bigram_df.columns


def test_empath_feature_extractor(setup_test_environment):
    output_folder = setup_test_environment / "test/data/feature_extracted_data"
    extractor = EmpathFeatureExtractor(output_folder=str(output_folder))

    # Test feature extraction
    extractor.extract_empath_features()

    # Assertions on extracted features
    assert extractor.features is not None
    assert not extractor.features.empty
    assert "label" in extractor.features.columns

    # Save features and validate file creation
    extractor.save_features_and_results(overwrite=True)
    empath_path = output_folder / "empath_features_with_labels.csv"
    assert empath_path.exists()

    # Validate saved CSV contents
    empath_df = pd.read_csv(empath_path)
    assert "label" in empath_df.columns
    assert empath_df.shape[0] > 0  # Ensure rows are present


def test_lda_feature_extraction_pipeline(setup_test_environment):
    output_folder = setup_test_environment / "test/data/feature_extracted_data"
    extractor = LDAFeatureExtractor(num_topics=2, output_folder=str(output_folder))

    # Run feature extraction pipeline
    extractor.run_feature_extraction()

    # Validate LDA model and outputs
    assert extractor.lda_model is not None
    lda_path = output_folder / "lda_topic_distributions_with_labels.csv"
    assert lda_path.exists()

    # Validate saved CSV contents
    lda_df = pd.read_csv(lda_path)
    assert "label" in lda_df.columns
    assert lda_df.shape[0] == 3  # 3 documents


def test_lda_feature_analysis_pipeline(setup_test_environment):
    output_folder = setup_test_environment / "test/data/feature_extracted_data"
    extractor = LDAFeatureExtractor(num_topics=2, output_folder=str(output_folder))

    # Run feature analysis pipeline
    extractor.run_feature_analysis()

    # Validate generated visualizations
    topic_table_path = output_folder / "topic_table_depressed.png"
    tsne_path = output_folder / "tsne_with_categories.png"
    assert topic_table_path.exists()
    assert tsne_path.exists()

    # Validate file size (ensure files are not empty)
    assert topic_table_path.stat().st_size > 0
    assert tsne_path.stat().st_size > 0
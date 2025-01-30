import os
import unittest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from model_training.model_training_func import ModelTrainer

class TestModelTraining(unittest.TestCase):

    def setUp(self):
        """Set up paths and test data for model training."""
        # Define the path for the features directory
        self.features_dir = "test/test_data/features"
        os.makedirs(self.features_dir, exist_ok=True)  # Ensure the directory exists

        # Create a test CSV file with more rows for stratification
        self.test_csv_files = [os.path.join(self.features_dir, "test_features.csv")]
        df = pd.DataFrame({
            "feature1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "feature2": [1, 0, 1, 0, 1, 0, 1, 0],
            "label": [1, 0, 1, 0, 1, 0, 1, 0],
        })
        df.to_csv(self.test_csv_files[0], index=False)

        # Pass the model class, not the instance
        self.model = RandomForestClassifier

    def test_load_and_combine_data(self):
        """Test that data loading and combining works correctly."""
        trainer = ModelTrainer(self.test_csv_files, self.model, "RandomForest")
        trainer.load_and_combine_data()
        self.assertIsNotNone(trainer.data, "Data is None after loading")
        self.assertEqual(len(trainer.data), 8, "Unexpected number of rows in combined data")

    def test_preprocess_data(self):
        """Test data preprocessing (scaling and splitting)."""
        trainer = ModelTrainer(self.test_csv_files, self.model, "RandomForest")
        trainer.load_and_combine_data()
        trainer.preprocess_data(test_size=0.25, pca_components=None)

        # Check train/test splits
        self.assertEqual(len(trainer.X_train), 6, "Unexpected size of training data")
        self.assertEqual(len(trainer.X_test), 2, "Unexpected size of test data")
        self.assertEqual(len(trainer.y_train), 6, "Unexpected size of training labels")
        self.assertEqual(len(trainer.y_test), 2, "Unexpected size of test labels")

    def test_train_model(self):
        """Test that the model training runs without errors."""
        trainer = ModelTrainer(self.test_csv_files, self.model, "RandomForest")
        trainer.load_and_combine_data()
        trainer.preprocess_data(test_size=0.25, pca_components=None)
        try:
            trainer.train_model()
        except Exception as e:
            self.fail(f"Model training failed with error: {e}")

    def test_evaluate_model(self):
        """Test that model evaluation metrics are calculated correctly."""
        trainer = ModelTrainer(self.test_csv_files, self.model, "RandomForest")
        trainer.load_and_combine_data()
        trainer.preprocess_data(test_size=0.25, pca_components=None)
        trainer.train_model()
        trainer.evaluate_model()

        # Check that metrics are not empty
        self.assertIn("Test Accuracy", trainer.metrics, "Test Accuracy not found in metrics")
        self.assertGreater(trainer.metrics["Test Accuracy"], 0, "Test Accuracy is 0 or less")

    def test_pipeline_execution(self):
        """Test the full pipeline execution (load, preprocess, train, evaluate)."""
        trainer = ModelTrainer(self.test_csv_files, self.model, "RandomForest")
        try:
            metrics = trainer.run_pipeline(pca_components=None)
            self.assertIn("Accuracy", metrics, "Pipeline did not return Accuracy")
            self.assertGreater(metrics["Accuracy"], 0, "Pipeline Accuracy is 0 or less")
        except Exception as e:
            self.fail(f"Pipeline execution failed with error: {e}")

    def tearDown(self):
        """Clean up test data."""
        for file in self.test_csv_files:
            if os.path.exists(file):
                os.remove(file)
        if os.path.exists(self.features_dir):
            os.rmdir(self.features_dir)  # Remove the directory if empty

if __name__ == "__main__":
    unittest.main()
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

###############################################################################
# ModelTrainer Class
#   - Loads & merges multiple CSV files into a single DataFrame.
#   - Expects the first CSV to contain the 'label' column.
#   - Trains a chosen ML model, evaluates, and can compare multiple models.
###############################################################################
class ModelTrainer:

    def __init__(self, csv_files, model, model_name, model_params=None, random_state=42):
        """
        csv_files: List of CSV paths (first CSV must have 'label').
        model: scikit-learn estimator class, e.g. RandomForestClassifier.
        model_name: String name for reporting.
        model_params: Dict of hyperparameters (optional).
        random_state: For reproducible splits.
        """
        self.csv_files = csv_files
        self.model_name = model_name
        self.model_class = model
        self.model_params = model_params if model_params else {}
        self.data = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.random_state = random_state
        self.metrics = {}
        self.scaler = StandardScaler()
        self.pca = None

    ###############################################################################
    # 1) Load & Combine Data
    #    - Merges features from multiple CSVs side by side
    #    - Uses 'label' from first CSV (drops 'label' in others)
    ###############################################################################
    def load_and_combine_data(self):
        data_frames = [pd.read_csv(file) for file in self.csv_files]
        labels = data_frames[0]['label']  # This is our primary label column

        # Remove label column from subsequent DataFrames
        for df in data_frames[1:]:
            if 'label' in df.columns:
                df.drop(columns=['label'], inplace=True)

        # Concatenate side-by-side
        combined_data = pd.concat(data_frames, axis=1)

        # Check size match
        if len(labels) != len(combined_data):
            raise ValueError("Mismatch between features and labels.")

        # Final combined set with label appended at end
        self.data = combined_data
        self.data['label'] = labels

    ###############################################################################
    # 2) Preprocess Data
    #    - Splits into Train/Test
    #    - Scales via StandardScaler
    #    - Optionally applies PCA for dimensionality reduction
    ###############################################################################
    def preprocess_data(self, test_size=0.2, pca_components=50):
        X = self.data.iloc[:, :-1]  # all cols except 'label'
        y = self.data['label']      # the 'label' column

        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        # Scale features to mean=0, std=1
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        # Apply PCA if requested
        if pca_components:
            self.pca = PCA(n_components=pca_components)
            self.X_train = self.pca.fit_transform(self.X_train)
            self.X_test = self.pca.transform(self.X_test)

    ###############################################################################
    # 3) Train the Model
    ###############################################################################
    def train_model(self):
        self.model = self.model_class(**self.model_params)
        self.model.fit(self.X_train, self.y_train)

    ###############################################################################
    # 4) Evaluate Model
    #    - Predict on test data, compute accuracy/f1/precision/recall
    #    - Also compute 10-fold CV accuracy on the training set
    ###############################################################################
    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=10, scoring='accuracy')

        self.metrics = {
            "Model": self.model_name,
            "Test Accuracy": accuracy,
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall,
            "CV Mean Accuracy": cv_scores.mean(),
            "CV Std Dev": cv_scores.std()
        }

    ###############################################################################
    # 5) Run the Entire Pipeline
    #    - Load, preprocess, train, evaluate
    #    - Returns main metrics in %
    ###############################################################################
    def run_pipeline(self, pca_components=None):
        self.load_and_combine_data()
        self.preprocess_data(pca_components=pca_components)
        self.train_model()
        self.evaluate_model()

        return {
            "Accuracy": round(self.metrics["Test Accuracy"] * 100, 2),
            "F1": round(self.metrics["F1 Score"] * 100, 2),
            "Precision": round(self.metrics["Precision"] * 100, 2),
            "Recall": round(self.metrics["Recall"] * 100, 2)
        }

    ###############################################################################
    # Utility: Save Results as an Image
    #    - Creates a DataFrame, plots as a table, saves as .png
    ###############################################################################
    @staticmethod
    def save_results_as_image(results, filename):
        df = pd.DataFrame(results)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(df.columns))))
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    ###############################################################################
    # Utility: Run & Save Results for Multiple Feature Sets & Models
    #    - feature_combinations: dict of {featureSetName: [list_of_csv_files]}
    #    - models_config: dict of {modelName: {"model_class":..., "params":...}}
    #    - For each combination of feature set & model, trains & collects metrics.
    #    - All results displayed & saved as a table image
    ###############################################################################
    @staticmethod
    def run_and_save_results(feature_combinations, models_config, output_filename, pca_components=None):
        results = []

        # For each named feature set
        for feature_name, feature_files in feature_combinations.items():
            print(f"\nTraining models for feature set: {feature_name}")
            feature_results = {"Feature": feature_name}

            # For each model in the config
            for model_name, config in models_config.items():
                print(f"  Training {model_name}...")
                trainer = ModelTrainer(
                    csv_files=feature_files,
                    model=config["model_class"],
                    model_name=model_name,
                    model_params=config["params"]
                )
                metrics = trainer.run_pipeline(pca_components=pca_components)
                # Combine Acc, F1, Precision, and Recall into a single string
                feature_results[model_name] = (
                    f"{metrics['Accuracy']}%/"
                    f"{metrics['F1']}%/"
                    f"{metrics['Precision']}%/"
                    f"{metrics['Recall']}%"
                )

            # Append row of results for this feature set
            results.append(feature_results)

        # Finally, save results as an image
        ModelTrainer.save_results_as_image(results, output_filename)
        print(f"\nResults table saved as {output_filename}")

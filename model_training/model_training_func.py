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
    def __init__(self, csv_files, model, model_name, model_params=None,
                 random_state=42, use_stratify=True):
        """
        Initialize the ModelTrainer.

        Args:
            csv_files (list[str]): Paths to CSV files (first must have 'label').
            model (class): A scikit-learn estimator class (e.g., RandomForestClassifier).
            model_name (str): Name/identifier for reporting.
            model_params (dict, optional): Hyperparameters for the estimator.
            random_state (int): Seed for reproducibility.
            use_stratify (bool): Whether to stratify the train/test split by label.
        """
        self.csv_files = csv_files
        self.model_name = model_name
        self.model_class = model
        self.model_params = model_params if model_params else {}
        self.random_state = random_state
        self.use_stratify = use_stratify

        # Internal placeholders
        self.data = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.metrics = {}
        self.scaler = StandardScaler()
        self.pca = None

    ###############################################################################
    # 1) Load & Combine Data
    #    - Merges features from multiple CSVs side by side
    #    - Uses 'label' from the first CSV (drops 'label' from subsequent ones)
    ###############################################################################
    def load_and_combine_data(self):
        try:
            data_frames = [pd.read_csv(file) for file in self.csv_files]
        except FileNotFoundError as e:
            raise FileNotFoundError(f"One or more files could not be found: {e}")
        except pd.errors.EmptyDataError as e:
            raise ValueError(f"One or more CSV files are empty: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error while reading files: {e}")

        if 'label' not in data_frames[0].columns:
            raise ValueError("The first CSV must contain a 'label' column.")

        labels = data_frames[0]['label']
        for df in data_frames[1:]:
            if 'label' in df.columns:
                df.drop(columns=['label'], inplace=True)

        combined_data = pd.concat(data_frames, axis=1)

        if len(labels) != len(combined_data):
            raise ValueError("Mismatch between features and labels row counts.")

        self.data = combined_data
        self.data['label'] = labels

    ###############################################################################
    # 2) Preprocess Data
    #    - Splits into Train/Test
    #    - Scales via StandardScaler
    #    - Optionally applies PCA for dimensionality reduction
    ###############################################################################
    def preprocess_data(self, test_size=0.2, pca_components=50):
        """
        Splits data into train/test sets, then applies scaling and optional PCA.

        Args:
            test_size (float): Fraction of data to use for testing.
            pca_components (int): Number of PCA components; if 0 or None, skip PCA.
        """
        X = self.data.iloc[:, :-1]  # All columns except the last ('label')
        y = self.data['label']

        # If requested, stratify by label so class distribution remains consistent
        stratify_target = y if self.use_stratify else None

        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=self.random_state,
                stratify=stratify_target
            )
        except ValueError as e:
            # Fallback to non-stratified split if stratification fails
            print(f"Warning: Stratified split failed. Using non-stratified split. Error: {e}")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=self.random_state
            )

        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        # PCA dimensionality reduction if desired
        if pca_components and pca_components > 0:
            self.pca = PCA(n_components=pca_components, random_state=self.random_state)
            self.X_train = self.pca.fit_transform(self.X_train)
            self.X_test = self.pca.transform(self.X_test)


    ###############################################################################
    # 3) Train the Model
    ###############################################################################
    def train_model(self):
        try:
            self.model = self.model_class(**self.model_params)
            self.model.fit(self.X_train, self.y_train)
        except TypeError as e:
            raise TypeError(f"Invalid model parameters: {e}")
        except ValueError as e:
            raise ValueError(f"Error during model training: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during training: {e}")


    ###############################################################################
    # 4) Evaluate Model
    #    - Predict on test data, compute accuracy/f1/precision/recall
    #    - Also compute 10-fold CV accuracy on the training set
    ###############################################################################
    def evaluate_model(self):
        try:
            y_pred = self.model.predict(self.X_test)
        except ValueError as e:
            raise ValueError(f"Error during prediction: {e}")

        try:
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
        except ValueError as e:
            raise ValueError(f"Error calculating metrics: {e}")

        try:
            # Dynamically determine n_splits based on the smallest class size
            min_class_size = min(self.y_train.value_counts().min(), len(self.X_train))
            n_splits = min(10, min_class_size)
            if n_splits < 2:
                raise ValueError("Not enough data points for cross-validation (n_splits must be >= 2).")

            cv_scores = cross_val_score(
                self.model, self.X_train, self.y_train, cv=n_splits, scoring='accuracy', n_jobs=-1
            )
        except ValueError as e:
            raise ValueError(f"Cross-validation error: {e}")

        self.metrics = {
            "Model": self.model_name,
            "Test Accuracy": accuracy,
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall,
            "CV Mean Accuracy": cv_scores.mean() if n_splits >= 2 else None,
            "CV Std Dev": cv_scores.std() if n_splits >= 2 else None
        }

    ###############################################################################
    # 5) Run the Entire Pipeline
    #    - Load, preprocess, train, evaluate
    #    - Returns main metrics in %
    ###############################################################################
    def run_pipeline(self, pca_components=None):
        """
        High-level method to load data, preprocess, train, and evaluate in sequence.

        Args:
            pca_components (int or None): If provided, apply PCA to reduce feature dims.

        Returns:
            dict: Contains main metrics in percentage form for quick viewing.
        """
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
        """
        Render a list of dicts (rows) as a matplotlib table and save to a .png.

        Args:
            results (list[dict]): Each dict is one row of data to display.
            filename (str): Where to save the table image.
        """
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
    def run_and_save_results(feature_combinations, models_config, output_filename,
                             pca_components=None, use_stratify=True):
        """
        Iterate over multiple feature sets and models, run the pipeline,
        then save all results to a single table image.

        Args:
            feature_combinations (dict): 
                Keys = feature set names, Values = list of CSV file paths.
            models_config (dict):
                Keys = model names, Values = {"model_class":..., "params":...}.
            output_filename (str): Path of the .png file to store results table.
            pca_components (int or None): If not None, apply PCA with that many components.
            use_stratify (bool): Whether to stratify the train/test split for each run.
        """
        results = []

        for feature_name, feature_files in feature_combinations.items():
            print(f"\nTraining models for feature set: {feature_name}")
            feature_results = {"Feature": feature_name}

            for model_name, config in models_config.items():
                print(f"  Training {model_name}...")
                trainer = ModelTrainer(
                    csv_files=feature_files,
                    model=config["model_class"],
                    model_name=model_name,
                    model_params=config["params"],
                    use_stratify=use_stratify
                )
                metrics = trainer.run_pipeline(pca_components=pca_components)

                # Concatenate Accuracy/F1/Precision/Recall in one string
                feature_results[model_name] = (
                    f"{metrics['Accuracy']}%/"
                    f"{metrics['F1']}%/"
                    f"{metrics['Precision']}%/"
                    f"{metrics['Recall']}%"
                )

            results.append(feature_results)

        ModelTrainer.save_results_as_image(results, output_filename)
        print(f"\nResults table saved as '{output_filename}'")
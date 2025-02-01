# **Tadesse et al Data Science Replicability Project**

## **Overview**
This project replicates the study by Tadesse et al. (2019), *“Detection of Depression-Related Posts in Reddit Social Media Forum.”* We aim to classify depression-related posts on Reddit using machine learning techniques. The process involves **feature extraction**, **analysis**, and **model training**, with different methods such as **Empath features**, **N-grams**, and **LDA**.

Our replication findings, including key similarities and differences from the original work, are detailed in the **"Replication of Tadesse et al."** PDF file in this repository.

---

## **Project Workflow**
The workflow includes the following key steps:

1. **Data Preprocessing**:
   - Raw posts are scraped from subreddits (`r/depression`, `r/breastcancer`, and others).
   - Text is cleaned (e.g., removing usernames, special characters, stopwords).

2. **Feature Extraction**:
   - Three types of features are extracted:
     - **N-grams**: Unigram and bigram frequency features.
     - **Empath**: Psycholinguistic features using the Empath library.
     - **LDA**: Latent Dirichlet Allocation (LDA) topic modeling.

3. **Feature Analysis**:
   - Visualizations and correlation analysis (e.g., word clouds, correlation tables).

4. **Model Training**:
   - Machine learning models (e.g., SVM, Random Forest) are trained on the extracted features.
   - Model evaluation is performed using metrics such as **accuracy**, **F1 score**, **precision**, and **recall**.

5. **Testing**:
   - Unit tests are implemented for **data preprocessing**, **feature extraction**, and **model training**.

---

## **Repository Structure**
```plaintext
.
├── data/                          # Stores raw, preprocessed, and extracted feature data
│   ├── feature_analysis_output/    # Visuals and CSVs generated from feature analysis
│   ├── preprocessed_posts/         # Cleaned and preprocessed Reddit posts
│   ├── reddit_scraped_posts/       # Raw posts collected from various subreddits
│   └── data_preprocessing/         # Scripts for scraping and cleaning data
│       └── data_preprocessing.py
├── feature_extraction/             # Feature extraction methods
│   ├── feature_extraction.py       # Main script for feature extraction
│   ├── feature_extraction_func.py  # Defines functions for feature extraction
│   ├── ngram_feature_extraction.py # N-gram extraction methods
│   ├── lda_feature_extraction.py   # LDA topic modeling methods
│   └── empath_feature_extraction.py # Empath feature extraction and analysis
├── feature_analysis/               # Visualization and analysis scripts
│   └── feature_analysis.py
├── model_training/                 # Model training and evaluation scripts
│   ├── model_training.py           # Main model training script
│   └── model_training_func.py      # Defines functions for training models
├── test/                           # Unit tests for various components
│   ├── test_data_preprocessing.py  # Tests for data preprocessing
│   ├── test_feature_extraction.py  # Tests for feature extraction
│   └── test_model_training.py      # Tests for model training
├── outputs/                        # Outputs from analysis and model evaluation
├── pyproject.toml                  # Defines build requirements, dependencies, linting, etc.
├── tox.ini                         # Automates testing, linting, and builds
└── README.md

### **Main Components**

*   **data/**: Organized in stages (raw, preprocessed, feature-extracted data). Contains:
    
    *   **reddit\_scraped\_posts/**: Raw scraped data from Reddit.
        
    *   **preprocessed\_posts/**: Cleaned, preprocessed text.
        
    *   **data\_preprocessing/**: Scripts for scraping and preprocessing data.
        
*   **feature\_extraction/**: Implements multiple feature extraction methods:
    
    *   **ngram\_feature\_extraction.py**: Extracts unigrams, bigrams, and filters based on Pointwise Mutual Information (PMI).
        
    *   **lda\_feature\_extraction.py**: Prepares documents, trains LDA models, and generates topic distributions.
        
    *   **empath\_feature\_extraction.py**: Extracts psycholinguistic features and computes correlations.
        
*   **feature\_analysis/**: Analyzes extracted features, generating visualizations like word clouds and correlation tables.
    
*   **model\_training/**: Trains and evaluates ML models, saving performance metrics and results.
    
*   **test/**: Contains unit tests for:
    
    *   **test\_data\_preprocessing.py**: Tests preprocessing functions.
        
    *   **test\_feature\_extraction.py**: Tests N-gram, LDA, and Empath feature extraction.
        
    *   **test\_model\_training.py**: Tests model loading, training, and evaluation.
        

**Running the Project**
-----------------------

### **1\. Install Dependencies**

*   Ensure you have **Python 3.9+** installed.
    
*   bashCopyEditpip install -r requirements.txt
    

### **2\. Run Data Preprocessing**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   bashCopyEditpython data_preprocessing/data_preprocessing.py   `

*   Processes raw Reddit posts and saves cleaned text to data/preprocessed\_posts/.
    

### **3\. Run Feature Extraction**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   bashCopyEditpython feature_extraction/feature_extraction.py   `

*   Extracts **N-gram**, **Empath**, and **LDA** features.
    
*   Outputs CSV files (e.g., unigram\_features\_with\_labels.csv) in the feature output folder.
    

### **4\. Run Feature Analysis**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   bashCopyEditpython feature_analysis/feature_analysis.py   `

*   Generates visualizations like word clouds and saves them to outputs/.
    

### **5\. Train Machine Learning Models**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   bashCopyEditpython model_training/model_training.py   `

*   Loads feature CSVs, trains models, and evaluates performance.
    
*   Saves accuracy, F1 scores, and confusion matrices to outputs/.
    

### **6\. Run Tests**

*   Use **tox** to automate testing and linting:
    

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   bashCopyEdittox   `

**References**
--------------

*   **Paper**: M. M. Tadesse, H. Lin, B. Xu, and L. Yang, “Detection of Depression-Related Posts in Reddit Social Media Forum,” _IEEE Access_, vol. 7, pp. 44883–44893, 2019.
    
*   **Dependencies**:
    
    *   **NLTK**, **scikit-learn**: Text processing and machine learning.
        
    *   **Empath**: Psycholinguistic feature extraction.
        
    *   **Gensim**: LDA topic modeling.
        
    *   **Pytest**: For unit testing.
        
    *   **Ruff** and **Mypy**: For linting and static type checks.
### **Main Components**

*   **data/**: Organized in stages (raw, preprocessed, feature-extracted data). Contains:
    
    *   **reddit\_scraped\_posts/**: Raw scraped data from Reddit.
        
    *   **preprocessed\_posts/**: Cleaned, preprocessed text.
        
    *   **data\_preprocessing/**: Scripts for scraping and preprocessing data.
        
*   **feature\_extraction/**: Implements multiple feature extraction methods:
    
    *   **ngram\_feature\_extraction.py**: Extracts unigrams, bigrams, and filters based on Pointwise Mutual Information (PMI).
        
    *   **lda\_feature\_extraction.py**: Prepares documents, trains LDA models, and generates topic distributions.
        
    *   **empath\_feature\_extraction.py**: Extracts psycholinguistic features and computes correlations.
        
*   **feature\_analysis/**: Analyzes extracted features, generating visualizations like word clouds and correlation tables.
    
*   **model\_training/**: Trains and evaluates ML models, saving performance metrics and results.
    
*   **test/**: Contains unit tests for:
    
    *   **test\_data\_preprocessing.py**: Tests preprocessing functions.
        
    *   **test\_feature\_extraction.py**: Tests N-gram, LDA, and Empath feature extraction.
        
    *   **test\_model\_training.py**: Tests model loading, training, and evaluation.
        

**Running the Project**
-----------------------

### **1\. Install Dependencies**

*   Ensure you have **Python 3.9+** installed.
    
*   bashCopyEditpip install -r requirements.txt
    

### **2\. Run Data Preprocessing**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   bashCopyEditpython data_preprocessing/data_preprocessing.py   `

*   Processes raw Reddit posts and saves cleaned text to data/preprocessed\_posts/.
    

### **3\. Run Feature Extraction**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   bashCopyEditpython feature_extraction/feature_extraction.py   `

*   Extracts **N-gram**, **Empath**, and **LDA** features.
    
*   Outputs CSV files (e.g., unigram\_features\_with\_labels.csv) in the feature output folder.
    

### **4\. Run Feature Analysis**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   bashCopyEditpython feature_analysis/feature_analysis.py   `

*   Generates visualizations like word clouds and saves them to outputs/.
    

### **5\. Train Machine Learning Models**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   bashCopyEditpython model_training/model_training.py   `

*   Loads feature CSVs, trains models, and evaluates performance.
    
*   Saves accuracy, F1 scores, and confusion matrices to outputs/.
    

### **6\. Run Tests**

*   Use **tox** to automate testing and linting:
    

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   bashCopyEdittox   `

**References**
--------------

*   **Paper**: M. M. Tadesse, H. Lin, B. Xu, and L. Yang, “Detection of Depression-Related Posts in Reddit Social Media Forum,” _IEEE Access_, vol. 7, pp. 44883–44893, 2019.
    
*   **Dependencies**:
    
    *   **NLTK**, **scikit-learn**: Text processing and machine learning.
        
    *   **Empath**: Psycholinguistic feature extraction.
        
    *   **Gensim**: LDA topic modeling.
        
    *   **Pytest**: For unit testing.
        
    *   **Ruff** and **Mypy**: For linting and static type checks
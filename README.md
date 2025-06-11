# Sentiment-analysis 
Overview
This project implements two sentiment analysis pipelines to classify social media posts into Positive, Negative, or Neutral sentiments using natural language processing (NLP) and machine learning techniques. The implementations are provided in separate Jupyter Notebooks: Bow (3).ipynb (using Bag-of-Words and VADER) and TF_IDF (2).ipynb (using TF-IDF vectorization). Both use the same dataset (sentimentdataset.csv) containing 732 posts with associated metadata, such as timestamps, user information, and engagement metrics. The goal is to preprocess text data, extract features, and evaluate multiple machine learning models for sentiment classification.
Dataset

File: sentimentdataset.csv
Columns:
ID: Unique identifier for each post
Text: Social media post content (e.g., "Enjoying a beautiful day at the park!")
Sentiment (Label): Sentiment label (e.g., Positive, Negative, Neutral, or granular labels like "Elation")
Timestamp: Date and time of the post
User: Username of the poster
Source: Platform (e.g., Twitter, Instagram, Facebook)
Topic: Hashtags or topics associated with the post
Retweets: Number of retweets
Likes: Number of likes
Country: Country of origin
Year, Month, Day, Hour: Parsed timestamp components


Size: 732 rows, 14 columns
Unique Sentiments: 279 granular labels, mapped to 3 categories (Positive, Negative, Neutral)
Sources: 4 platforms (Twitter, Instagram, Facebook, others)
Diversity: Posts cover topics like nature, fitness, travel, and food, from multiple countries.

Requirements
To run the notebooks, install the following Python libraries:
pip install pandas numpy matplotlib seaborn nltk spacy gensim sklearn vaderSentiment

Additional setup:

Download NLTK data: stopwords, wordnet, punkt
Install SpaCy model: python -m spacy download en_core_web_sm

Implementations
1. Bag-of-Words Implementation (Bow (3).ipynb)
Project Structure

Data Loading and Exploration:

Load the dataset using pandas.
Drop ID and User columns to focus on relevant features.
Display initial rows, shape (732x14), data types, and unique values (e.g., 279 sentiments, 4 sources).
Perform basic data cleaning (e.g., handle missing values, if any).


Text Preprocessing:

Clean text by converting to lowercase, removing punctuation, URLs, and special characters using re and string.
Tokenize using nltk or spacy.
Remove stopwords from nltk.corpus.stopwords.
Apply stemming with SnowballStemmer or lemmatization with WordNetLemmatizer.
Extract features using TfidfVectorizer or CountVectorizer from sklearn.
Generate initial sentiment scores using VADER (vaderSentiment).


Sentiment Mapping:

Map 279 granular sentiment labels to 3 categories: 0 (Negative), 1 (Neutral), 2 (Positive).


Machine Learning Models:

Naive Bayes (MultinomialNB):
Train and evaluate using 5-fold cross-validation (KFold, random_state=42).
Cross-validation mean accuracy: ~0.799, standard deviation: ~0.009.
Test accuracy: ~0.789.
Strong F1-score for Neutral class (0.83).


Logistic Regression:
Train with random_state=0 for reproducibility.
Cross-validation mean accuracy: ~0.817, standard deviation: ~0.021.
Test accuracy: ~0.76.
High precision for Positive class (0.88), lower recall for Neutral (0.44).


Support Vector Machine (SVM):
Use linear kernel (SVC(kernel='linear')).
Cross-validation mean accuracy: ~0.829, standard deviation: ~0.021.
Test accuracy: ~0.796.
Balanced performance across all classes.




Evaluation:

Use 5-fold cross-validation to ensure robust performance estimates.
Report accuracy, precision, recall, and F1-score using classification_report.
Generate confusion matrices to analyze class-specific performance.
SVM outperforms other models due to its ability to handle high-dimensional text data effectively.



Results

Naive Bayes: Test accuracy ~0.789, excels in predicting Neutral class.
Logistic Regression: Test accuracy ~0.76, strong precision for Positive but weaker recall for Neutral.
SVM: Test accuracy ~0.796, best overall performance with balanced classification.

2. TF-IDF Implementation (TF_IDF (2).ipynb)
Project Structure

Data Loading and Exploration:

Load dataset with pandas.
Drop ID and User columns.
Verify dataset characteristics: 732 rows, 279 unique sentiments, 4 unique sources.
Explore sample texts and metadata.


Text Preprocessing:

Function: preprocess_text
Tokenize using nltk.word_tokenize.
Convert to lowercase, retain only alphabetic tokens.
Remove stopwords from nltk.corpus.stopwords.
Join tokens into a string.


Function: stem_text
Tokenize and stem using PorterStemmer.
Join stemmed tokens into a string.


Create Processed_Text by applying preprocess_text to Text.
Create Stemmed_Text by applying preprocess_text to Processed_Text (note: stem_text intended but not used correctly).
Vectorize Stemmed_Text and Topic columns separately using TfidfVectorizer.
Concatenate TF-IDF features from Stemmed_Text and Topic into a single feature matrix.


Sentiment Mapping:

Map granular sentiment labels to sentiment_category (y): 0 (Negative), 1 (Neutral), 2 (Positive).


Machine Learning Models:

Naive Bayes (MultinomialNB):
Cross-validation (5-fold, random_state=42): Mean accuracy ~0.842, standard deviation ~0.024.
Test accuracy: ~0.789.
Consistent performance across classes.


Logistic Regression:
Configured with random_state=0.
Cross-validation: Mean accuracy ~0.761, standard deviation ~0.058.
Test accuracy not explicitly reported in the notebook.
Struggles with class imbalance.


Support Vector Machine (SVM):
Linear kernel (SVC(kernel='linear')).
Cross-validation: Mean accuracy ~0.837, standard deviation ~0.018.
Test accuracy: ~0.810.
Best performance due to effective handling of TF-IDF features.


Random Forest Classifier:
Defined in a pipeline with CountVectorizer (max_features=4) but not fully evaluated.




Evaluation:

Perform 5-fold cross-validation (KFold, random_state=42).
Report accuracy, confusion matrix, and classification report for Naive Bayes and SVM.
SVM achieves the highest test accuracy (~0.810).
Prediction code for new inputs contains errors (e.g., vectorizer.transform expects an iterable, not a string).



Results

Naive Bayes: Test accuracy ~0.789, consistent across classes.
Logistic Regression: Cross-validation mean accuracy ~0.761, limited by class imbalance.
SVM: Test accuracy ~0.810, best performance with balanced classification.
Random Forest: Not fully evaluated, requires further implementation.

Notes

Combining TF-IDF features from Stemmed_Text and Topic enhances feature richness compared to text-only features.
The Stemmed_Text column is incorrectly generated by reapplying preprocess_text instead of stem_text, reducing stemming effectiveness.
Prediction code errors (e.g., ValueError: Iterable over raw text documents expected) can be fixed by passing a list, e.g., vectorizer.transform([pross_x]).

Usage

Clone the repository or download the notebooks and dataset.
Ensure all dependencies are installed.
Place sentimentdataset.csv in the same directory as the notebooks.
Open the notebooks in Jupyter Notebook or Google Colab:
Run Bow (3).ipynb sequentially to reproduce the Bag-of-Words analysis.
Run TF_IDF (2).ipynb sequentially to reproduce the TF-IDF analysis, noting that the prediction code may require debugging.


Compare model performance across the two implementations.

Comparative Analysis

Feature Extraction:
Bow (3).ipynb: Uses TfidfVectorizer or CountVectorizer with VADER sentiment scores for additional context.
TF_IDF (2).ipynb: Uses TfidfVectorizer on both Stemmed_Text and Topic, providing richer features by incorporating hashtag/topic data.


Model Performance:
SVM: Best in both implementations (Bow: ~0.796 test accuracy; TF_IDF: ~0.810 test accuracy), likely due to its ability to handle high-dimensional data.
Naive Bayes: Consistent performance (~0.789 test accuracy in both), suitable for text classification.
Logistic Regression: Better in Bow (0.76 test accuracy) than TF_IDF (0.761 cross-validation), impacted by class imbalance in TF_IDF.
Random Forest: Only partially implemented in TF_IDF, not evaluated.


Preprocessing:
Bow: Offers flexible stemming/lemmatization options, integrates VADER scores.
TF_IDF: Focuses on TF-IDF with stemming but has redundant preprocessing (e.g., Stemmed_Text not properly stemmed).


Scalability:
TF_IDF: Potentially more scalable due to inclusion of topic features, suitable for larger datasets.
Bow: Simpler and faster for smaller datasets, easier to implement.


Robustness:
Bow: More robust preprocessing pipeline with VADER integration.
TF_IDF: Richer features but requires fixing prediction code and stemming logic.



Notes

Class Imbalance: The Neutral class dominates in the dataset, impacting model performance (e.g., lower recall for Negative in Logistic Regression).
Preprocessing Impact: Thorough text cleaning (stopword removal, stemming/lemmatization) and TF-IDF vectorization significantly improve accuracy in both implementations.
Errors in TF_IDF: The prediction code fails due to incorrect input format for vectorizer.transform. The Stemmed_Text column misses proper stemming, which could be improved by applying stem_text.

Future Improvements

General Improvements:
Experiment with advanced models (e.g., Random Forest, XGBoost, or deep learning models like LSTM, BERT).
Incorporate word embeddings (e.g., Word2Vec, GloVe) for better semantic representation.
Address class imbalance using techniques like SMOTE or oversampling.
Perform hyperparameter tuning with grid search or random search.
Expand the dataset with more diverse sources, languages, or larger sample sizes.


Bow (3).ipynb Specific:
Integrate topic features (as in TF_IDF) to enhance feature set.
Explore weighting VADER scores in the model input for improved sentiment detection.


TF_IDF (2).ipynb Specific:
Fix the prediction code to handle single inputs correctly (e.g., use vectorizer.transform([text])).
Correctly apply stem_text for Stemmed_Text to leverage stemming benefits.
Fully implement and evaluate the Random Forest pipeline.
Optimize TF-IDF parameters (e.g., max_features, ngram_range) for better performance.



License
This project is for educational purposes. Ensure proper attribution if used or modified.

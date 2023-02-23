import pandas as pd
import nltk
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
nltk.download('wordnet')

# Load the dataset
df = pd.read_csv("new_train.csv")

# Define the parameter grid to search over
param_grid = {
    "vectorizer__max_df": [0.2, 0.3, 0.5],
    "vectorizer__min_df": [17, 21, 23],
    "vectorizer__ngram_range": [(1, 1), (1,2)],
    "clf__alpha": [0.01, 0.1, 0.5, 1]
}

# Define the pipeline
pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(stop_words="english")),
    ("clf", MultinomialNB())
])

# Define the grid search
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, n_jobs=-1, scoring='f1_macro')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df["transcription"], df["medical_specialty"], test_size=0.2, random_state=42)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Print the best parameter combination and accuracy
print("Best parameters:", grid_search.best_params_)
print("Accuracy:", grid_search.best_score_)
print("Best F1 score:", grid_search.cv_results_['mean_test_score'][grid_search.best_index_])

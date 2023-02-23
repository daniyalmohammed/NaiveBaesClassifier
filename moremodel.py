import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer

# Load the CSV file as a pandas DataFrame
df = pd.read_csv('new_train.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['transcription'], df['medical_specialty'], test_size=0.2, random_state=42)

# Define the pipeline for the model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

# Define the hyperparameters to search over using GridSearchCV
param_grid = {
    'tfidf__ngram_range': [(1, 1)],
    'clf__C': [1]
}

# Define the scoring metric to optimize for
scorer = make_scorer(f1_score, average='macro')

# Perform a grid search over the hyperparameters using cross-validation
grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring=scorer, cv=5)
grid_search.fit(X_train, y_train)

# Evaluate the best model on the test set
y_pred = grid_search.best_estimator_.predict(X_test)
f1_macro = f1_score(y_test, y_pred, average='macro')
print("Best F1-macro score:", f1_macro)

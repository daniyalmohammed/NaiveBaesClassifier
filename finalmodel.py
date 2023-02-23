# Import necessary libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv("new_train.csv")

# Preprocess the data
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.2, min_df=21, ngram_range=(1, 2))
X = vectorizer.fit_transform(df["transcription"])
y = df["medical_specialty"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf = MultinomialNB(alpha=0.01)
clf.fit(X_train, y_train)

# Test the model
y_pred = clf.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

from sklearn.metrics import f1_score
macro_f1 = f1_score(y_test, y_pred, average='macro')
micro_f1 = f1_score(y_test, y_pred, average='micro')
print("Macro F1-score: {:.2f}".format(macro_f1))
print("Micro F1-score: {:.2f}".format(micro_f1))

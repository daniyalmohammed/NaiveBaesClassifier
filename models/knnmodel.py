# KNN Model - 0.24
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

# Load data into a Pandas DataFrame
df = pd.read_csv('new_train.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['transcription'], df['medical_specialty'], test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a KNN model
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train_vec, y_train)

# Evaluate the model
score = clf.score(X_test_vec, y_test)
print('Accuracy:', score)
# Multinomial Model - 0.3 accuracy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv("new_train.csv")

# Preprocess the data
vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["transcription"])
y = df["medical_specialty"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=21)

# Train the model
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Test the model
y_pred = clf.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

# Predict the specialty of a new transcription and find confidence of prediction
new_transcription = ""
new_transcription_vec = vectorizer.transform([new_transcription])
prediction = clf.predict(new_transcription_vec)
predicted_prob = clf.predict_proba(new_transcription_vec)

# Print the predicted specialty
print("Predicted specialty:", prediction)

print("Predictied probabilities: ", predicted_prob)
print("Max Guess: ", predicted_prob.max())


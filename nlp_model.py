import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import _stop_words
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import classification_report
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

## Reading in our data
df = pd.read_csv("./IntactInstructions/new_train.csv", index_col=0)
print("Test size with duplicates: ", len(df))

# Create labels/target values
y = df.labels
print("Label size: ", len(y))
y

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df["transcription"], y, test_size=0.6, random_state=42)

# X_train: training data of features
print("X_train size: ", len(X_train))
# y_train: training data of label
print("y_train size: ", len(y_train))

# X_test: test data of features
print("X_test size: ", len(X_test))
# y_test: test data of label
print("y_test size: ", len(y_test))

# X_train
# y_train[:50]
# X_test
# y_test



# Instantiate the WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()


# Custom pre-processing function
def preprocess_data(text):
    text = text.lower()
    text = re.sub(r'\d|_', '', text) # removes digits and '_'
    text = wordnet_lemmatizer.lemmatize(text)
    return text

# , preprocessor=preprocess_data
# Initialize a CountVectorizer object
count_vectorizer = CountVectorizer(stop_words="english", preprocessor=preprocess_data, max_df=0.2, min_df=20, ngram_range=(1, 1))

print(type(count_vectorizer))



# Fit and transform the TRAINING data using only the 'transciption' column values
count_train = count_vectorizer.fit_transform(X_train.values)
# Transform the TEST data using only the 'transciption' column values
count_test = count_vectorizer.transform(X_test.values)


# Print number of words processing
print("Number of words: ", len(count_vectorizer.get_feature_names_out())) # number of test data from split
# Print the features (individual tokens) of the count_vectorizer
print(count_vectorizer.get_feature_names_out()[:500])

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

# Load the CSV file into a pandas dataframe
df1 = pd.read_csv('./IntactInstructions/new_train.csv')

# Split the data into training and testing sets
X_train1, X_test1, y_train1, y_test1 = train_test_split(df['transcription'], df['medical_specialty'], test_size=0.2, random_state=42)

# Create a CountVectorizer to convert the text into a bag-of-words representation
vectorizer1 = CountVectorizer(max_features=10000)
X_train_vectors1 = vectorizer1.fit_transform(X_train1)
X_test_vectors1 = vectorizer1.transform(X_test1)

# Define the hyperparameters to tune for each classifier
nb_params = {'alpha': [0.1, 0.5, 1.0]}
#svm_params = {'C': [0.1, 0.5, 1.0]}
dt_params = {'max_depth': [5, 10, 15], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
#gb_params = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15], 'learning_rate': [0.1, 0.5, 1.0]}
#rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
knn_params = {'n_neighbors': [3, 5, 7]}
#mlp_params = {'hidden_layer_sizes': [(100,), (100, 50), (100, 50, 25)], 'alpha': [0.0001, 0.001, 0.01]}
#xgb_params = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15], 'learning_rate': [0.1, 0.5, 1.0]}

# Train and evaluate each classifier using grid search and cross-validation
classifiers = [('Naive Bayes', MultinomialNB(), nb_params),
               #('Support Vector Machines', LinearSVC(), svm_params),
               ('Decision Tree', DecisionTreeClassifier(), dt_params),
               #('Gradient Boosting', GradientBoostingClassifier(), gb_params),
               #('Random Forest', RandomForestClassifier(), rf_params),
               ('K-Nearest Neighbors', KNeighborsClassifier(), knn_params),
               #('Multi-Layer Perceptron', MLPClassifier(), mlp_params),
               #('XGBoost', XGBClassifier(), xgb_params)
               ]

for name, clf, params in classifiers:
    print(f'starting... {name}')
    grid_search = GridSearchCV(clf, params, cv=5, n_jobs=-1, scoring='f1_macro')
    grid_search.fit(X_train_vectors1, y_train1)
    y_pred1 = grid_search.predict(X_test_vectors1)
    f1_macro = f1_score(y_test1, y_pred1, average='macro')
    print(f'Best parameters: {grid_search.best_params_}')
    print(f'Testing F1-macro score: {f1_macro:.3f}')
    print('-' * 80)



# Instantiate a Multinomial Naive Bayes classifier
nb_clf = MultinomialNB(alpha=0.4)
# Fit the classifier to the training data
nb_clf.fit(count_train, y_train)
# Create the predicted tags
pred = nb_clf.predict(count_test)

# Print the predictions for each row of the dataset (1001 rows)
print("Number of predictions: ", len(pred)) # Equal to the number of test data (when it got split)
print(pred)



# Calculate the accuracy score
score = metrics.accuracy_score(y_test, pred)

print(score)
print(classification_report(y_test, pred))


pred = nb_clf.predict(df['transcription'])
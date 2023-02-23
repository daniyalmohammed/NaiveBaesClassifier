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

# Load the CSV file into a pandas dataframe
df = pd.read_csv('new_train.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['transcription'], df['medical_specialty'], test_size=0.2, random_state=42)

# Create a CountVectorizer to convert the text into a bag-of-words representation
vectorizer = CountVectorizer(max_features=10000)
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Define the hyperparameters to tune for each classifier
nb_params = {'alpha': [0.1, 0.5, 1.0]}
#svm_params = {'C': [0.1, 0.5, 1.0]}
#dt_params = {'max_depth': [5, 10, 15], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
#gb_params = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15], 'learning_rate': [0.1, 0.5, 1.0]}
#rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
knn_params = {'n_neighbors': [3, 5, 7]}
#mlp_params = {'hidden_layer_sizes': [(100,), (100, 50), (100, 50, 25)], 'alpha': [0.0001, 0.001, 0.01]}
xgb_params = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15], 'learning_rate': [0.1, 0.5, 1.0]}

# Train and evaluate each classifier using grid search and cross-validation
classifiers = [('Naive Bayes', MultinomialNB(), nb_params),
               #('Support Vector Machines', LinearSVC(), svm_params),
               #('Decision Tree', DecisionTreeClassifier(), dt_params),
               #('Gradient Boosting', GradientBoostingClassifier(), gb_params),
               #('Random Forest', RandomForestClassifier(), rf_params),
               ('K-Nearest Neighbors', KNeighborsClassifier(), knn_params),
               #('Multi-Layer Perceptron', MLPClassifier(), mlp_params),
               ('XGBoost', XGBClassifier(), xgb_params)]

for name, clf, params in classifiers:
    print(f'starting... {name}')
    grid_search = GridSearchCV(clf, params, cv=5, n_jobs=-1, scoring='f1_macro')
    grid_search.fit(X_train_vectors, y_train)
    y_pred = grid_search.predict(X_test_vectors)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    print(f'Best parameters: {grid_search.best_params_}')
    print(f'Testing F1-macro score: {f1_macro:.3f}')
    print('-' * 80)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import json
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

# Load the preprocessed data
with open('../output/preprocessed_data.json', 'r') as f:
    data = [json.loads(line) for line in f]

df = pd.DataFrame(data)

# Convert lists back to numpy arrays
df['mfcc'] = df['mfcc'].apply(lambda x: np.array(x))
df['chroma'] = df['chroma'].apply(lambda x: np.array(x))
df['spectral_contrast'] = df['spectral_contrast'].apply(lambda x: np.array(x))

# Prepare the feature matrix (X) and target vector (y)
X = np.hstack([
    np.vstack(df['mfcc'].values),
    np.vstack(df['chroma'].values),
    np.vstack(df['spectral_contrast'].values)
])
y = df['emotion']

# Encode the labels (y)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Feature normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# RandomizedSearchCV for Random Forest
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

random_search = RandomizedSearchCV(RandomForestClassifier(), param_distributions=param_dist, n_iter=20, cv=3, n_jobs=-1, verbose=2)
random_search.fit(X_train, y_train)

# Evaluate the best RandomForest model
best_rf = random_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"RandomizedSearchCV Random Forest Accuracy: {accuracy_rf:.4f}")

# Bayesian Optimization with BayesSearchCV for Random Forest
search_spaces = {
    'n_estimators': Integer(100, 500),
    'max_depth': Integer(5, 50),
    'min_samples_split': Integer(2, 10),
    'min_samples_leaf': Integer(1, 5)
}

opt = BayesSearchCV(
    estimator=RandomForestClassifier(),
    search_spaces=search_spaces,
    n_iter=32,
    cv=3,
    n_jobs=-1,
    verbose=2
)

opt.fit(X_train, y_train)

# Evaluate the best Bayesian Optimized Random Forest model
best_rf_bayes = opt.best_estimator_
y_pred_bayes = best_rf_bayes.predict(X_test)
accuracy_bayes = accuracy_score(y_test, y_pred_bayes)
print(f"Bayesian Optimized Random Forest Accuracy: {accuracy_bayes:.4f}")

# Ensemble Method: Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('rf', best_rf),
        ('svc', SVC(probability=True)),
        ('gb', GradientBoostingClassifier())
    ], 
    voting='soft'
)

# Fit the voting classifier
voting_clf.fit(X_train, y_train)

# Evaluate the voting classifier
y_pred_voting = voting_clf.predict(X_test)
accuracy_voting = accuracy_score(y_test, y_pred_voting)
print(f"Voting Classifier Accuracy: {accuracy_voting:.4f}")


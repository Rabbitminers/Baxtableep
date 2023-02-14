import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load datase
df = pd.read_csv('data/data.csv')

# Split into text and labels
text = df['text'].astype(str)
labels = df['is_offensive']

# Convert text into bag-of-words representation
vectorizer = CountVectorizer(stop_words='english', min_df=0.0001)
X = vectorizer.fit_transform(text)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Evaluate performance on testing set
y_pred = clf.predict(X_test)

# Save the trained model using joblib
joblib.dump(clf, 'data/model.joblib')
joblib.dump(vectorizer, 'data/vec.joblib')

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred))
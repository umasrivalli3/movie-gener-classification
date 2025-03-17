import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset
data = {
    "Plot": [
        "A young boy discovers he has magical powers and attends a school of wizardry.",
        "A group of space rebels fight against an evil empire in a galaxy far away.",
        "A detective investigates a series of mysterious murders in a small town.",
        "A scientist invents a time machine and goes on an adventure in the past and future.",
        "A couple falls in love despite belonging to rival families in medieval times."
    ],
    "Genre": ["Fantasy", "Sci-Fi", "Mystery", "Sci-Fi", "Romance"]
}

df = pd.DataFrame(data)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

df["Cleaned_Plot"] = df["Plot"].apply(preprocess_text)

# Convert text data into TF-IDF features
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(df["Cleaned_Plot"])
y = df["Genre"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predict and evaluate
y_pred = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
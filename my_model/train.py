import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, classification_report

#load the 20 newsgroups dataset
train_data = load_files("model/aclImdb/train", categories=["neg", "pos"])
test_data = load_files("model/aclImdb/test", categories=["neg", "pos"])

# create a dataframe from the dataset
df_train = pd.DataFrame({'text': train_data.data, 'label': train_data.target})
df_test = pd.DataFrame({'text': test_data.data, 'label': test_data.target})

#print(df_train.head())
#print(df_test.head())
# map target labels to target names
df_train['label'] = df_train['label'].map(lambda x: train_data.target_names[x])
df_test['label'] = df_test['label'].map(lambda x: test_data.target_names[x])

X_train = df_train['text']
X_test = df_test['text']
y_train = df_train['label']
y_test = df_test['label']
#y = df['label']

# split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#feature extraction
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Naive Bayes Classifier
classifier = LogisticRegression()
classifier.fit(X_train_vec, y_train)

# predictions of trained model
predictions = classifier.predict(X_test_vec)
joblib.dump(classifier, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

#my_text = input("entrez une note en anglais please:\n")
#text_vec = vectorizer.transform([my_text])
#text_result = classifier.predict(text_vec)
#text_proba = classifier.predict_proba(text_vec)
#print(f"it is in the class {text_result[0]} with a probability of {text_proba[0][1]}")

# model performance
#accuracy = accuracy_score(y_test, predictions)
#print(f'Accuracy: {accuracy:.2f}')
#report = classification_report(y_test, predictions)
#print('Classification Report:\n', report)
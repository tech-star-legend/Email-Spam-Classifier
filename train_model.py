import pandas as pd
import numpy as np
import pickle
import warnings

warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud

# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Preprocessing function
from utils.preprocess import transform_text

# -------------------------------
# LOAD DATASET
# -------------------------------

print("\nLoading Dataset...\n")

df = pd.read_csv('spam.csv', encoding='latin-1')

# Keep required columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']

print("Dataset Loaded Successfully!\n")

# -------------------------------
# DATA CLEANING
# -------------------------------

print("Cleaning Dataset...\n")

# Remove duplicates
df.drop_duplicates(inplace=True)

# Convert labels
df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

print("Dataset Cleaned Successfully!\n")

# -------------------------------
# NLP PREPROCESSING
# -------------------------------

print("Applying NLP Processing...\n")

df['transformed_message'] = df['message'].apply(transform_text)

print("NLP Processing Completed!\n")

# -------------------------------
# TF-IDF VECTORIZATION
# -------------------------------

print("Performing TF-IDF Vectorization...\n")

tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(df['transformed_message']).toarray()

y = df['label'].values

print("TF-IDF Completed!\n")

# -------------------------------
# TRAIN TEST SPLIT
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------------
# MODELS
# -------------------------------

models = {

    "Naive Bayes": MultinomialNB(),

    "Logistic Regression": LogisticRegression(),

    "SVM": SVC(kernel='linear', probability=True)
}

results = []

# -------------------------------
# TRAINING LOOP
# -------------------------------

print("Training Models...\n")

best_model = None
best_accuracy = 0
best_model_name = ""

for name, model in models.items():

    print(f"Training {name}...\n")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])

    print(f"{name} Results:")
    print("-" * 40)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

    print("\nClassification Report:\n")

    print(classification_report(y_test, y_pred))

    print("\n")

    # Save best model
    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_model = model
        best_model_name = name

# -------------------------------
# SAVE MODELS
# -------------------------------

print(f"Best Model: {best_model_name}")
print(f"Best Accuracy: {round(best_accuracy, 4)}\n")

pickle.dump(tfidf, open('model/vectorizer.pkl', 'wb'))

pickle.dump(
    models['Naive Bayes'],
    open('model/naive_bayes.pkl', 'wb')
)

pickle.dump(
    models['Logistic Regression'],
    open('model/logistic.pkl', 'wb')
)

pickle.dump(
    models['SVM'],
    open('model/svm.pkl', 'wb')
)

print("All Models Saved Successfully!\n")

# -------------------------------
# RESULTS DATAFRAME
# -------------------------------

results_df = pd.DataFrame(
    results,
    columns=[
        'Model',
        'Accuracy',
        'Precision',
        'Recall',
        'F1 Score'
    ]
)

print(results_df)

# -------------------------------
# VISUALIZATION SECTION
# -------------------------------

print("\nGenerating Graphs...\n")

# Accuracy Comparison Chart
plt.figure(figsize=(8, 5))

sns.barplot(
    x='Model',
    y='Accuracy',
    data=results_df
)

plt.title('Model Accuracy Comparison')

plt.ylabel('Accuracy')

plt.savefig('static/charts/accuracy.png')

plt.close()

# Dataset Distribution
plt.figure(figsize=(6, 6))

df['label'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    labels=['Ham', 'Spam']
)

plt.title('Dataset Distribution')

plt.ylabel('')

plt.savefig('static/charts/dataset_distribution.png')

plt.close()

# Confusion Matrix for Best Model
y_pred_best = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title(f'Confusion Matrix ({best_model_name})')

plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.savefig('static/charts/confusion_matrix.png')

plt.close()

# WordCloud for Spam Messages
spam_words = df[df['label'] == 1]['transformed_message']

wc = WordCloud(
    width=800,
    height=400,
    background_color='white'
)

spam_wc = wc.generate(" ".join(spam_words))

plt.figure(figsize=(10, 5))

plt.imshow(spam_wc)

plt.axis('off')

plt.title('Important Spam Keywords')

plt.savefig('static/charts/spam_words.png')

plt.close()

print("Graphs Generated Successfully!\n")

print("=" * 50)
print("PHASE 1 COMPLETED SUCCESSFULLY")
print("=" * 50)
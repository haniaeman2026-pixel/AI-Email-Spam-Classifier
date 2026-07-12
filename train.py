# ==========================================================
# Email Spam Classifier - Train Model
# Developed by: Hania Eman
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import re
import string
import joblib
import pandas as pd

from datasets import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

print("=" * 60)
print("        EMAIL SPAM CLASSIFIER - TRAINING STARTED")
print("=" * 60)


# ==========================================================
# STEP 1 - LOAD DATASET
# ==========================================================

print("\nDownloading dataset from Hugging Face...")

dataset = load_dataset("ucirvine/sms_spam")

train_data = dataset["train"]

df = pd.DataFrame(train_data)

print("\nDataset Loaded Successfully!")
print(df.head())

print("\nDataset Shape:", df.shape)


# ==========================================================
# STEP 2 - RENAME COLUMNS
# ==========================================================

df.rename(columns={
    "sms": "message",
    "label": "label"
}, inplace=True)

print("\nColumns:")
print(df.columns)


# ==========================================================
# STEP 3 - CHECK MISSING VALUES
# ==========================================================

print("\nMissing Values")
print(df.isnull().sum())

df.dropna(inplace=True)

df.drop_duplicates(inplace=True)

print("\nDataset Shape After Cleaning:", df.shape)


# ==========================================================
# STEP 4 - TEXT CLEANING FUNCTION
# ==========================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\\S+", "", text)

    text = re.sub(r"www\\S+", "", text)

    text = re.sub(r"\\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\\s+", " ", text).strip()

    return text


print("\nCleaning Text...")

df["message"] = df["message"].apply(clean_text)

print("Text Cleaning Completed.")


# ==========================================================
# STEP 5 - FEATURES & LABELS
# ==========================================================

X = df["message"]

y = df["label"]

print("\nTotal Samples :", len(X))


# ==========================================================
# STEP 6 - TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))


# ==========================================================
# STEP 7 - TF-IDF VECTORIZATION
# ==========================================================

print("\nCreating TF-IDF Features...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train = vectorizer.fit_transform(X_train)

X_test = vectorizer.transform(X_test)

print("Feature Extraction Completed!")

print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)


# ==========================================================
# SAVE VECTORIZER
# ==========================================================

joblib.dump(vectorizer, "vectorizer.pkl")

print("\nVectorizer Saved Successfully!")
print("\nReady For Model Training...")
# ==========================================================
# STEP 8 - TRAIN MACHINE LEARNING MODEL
# ==========================================================

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("\n" + "=" * 60)
print("TRAINING MODEL...")
print("=" * 60)

model = MultinomialNB()

model.fit(X_train, y_train)

print("Model Training Completed Successfully!")


# ==========================================================
# STEP 9 - MAKE PREDICTIONS
# ==========================================================

print("\nMaking Predictions...")

y_pred = model.predict(X_test)

print("Prediction Completed!")


# ==========================================================
# STEP 10 - MODEL ACCURACY
# ==========================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print(f"MODEL ACCURACY : {accuracy * 100:.2f}%")
print("=" * 60)


# ==========================================================
# STEP 11 - CLASSIFICATION REPORT
# ==========================================================

print("\nClassification Report\n")

print(classification_report(
    y_test,
    y_pred,
    digits=4
))


# ==========================================================
# STEP 12 - CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix\n")

print(cm)


# ==========================================================
# STEP 13 - SAVE TRAINED MODEL
# ==========================================================

joblib.dump(model, "spam_model.pkl")

print("\nModel Saved Successfully!")

print("File Name : spam_model.pkl")


# ==========================================================
# STEP 14 - TEST SAMPLE
# ==========================================================

print("\nTesting Model With Sample Email...\n")

sample_email = """
Congratulations!

You have won a FREE iPhone.

Click here now to claim your prize.
"""

sample_email = clean_text(sample_email)

sample_vector = vectorizer.transform([sample_email])

prediction = model.predict(sample_vector)[0]

probability = model.predict_proba(sample_vector)[0]

confidence = probability.max() * 100


print("-" * 60)

print("Sample Email :")
print(sample_email)

print("-" * 60)

if prediction == 1:
    print("Prediction : SPAM")
else:
    print("Prediction : NOT SPAM")

print(f"Confidence : {confidence:.2f}%")

print("-" * 60)


# ==========================================================
# TRAINING COMPLETED
# ==========================================================

print("\n" + "=" * 60)
print("EMAIL SPAM CLASSIFIER TRAINING COMPLETED")
print("=" * 60)

print("\nGenerated Files")

print("✔ spam_model.pkl")
print("✔ vectorizer.pkl")

print("\nNow Run Streamlit App.")
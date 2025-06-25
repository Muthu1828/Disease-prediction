import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # ✅ Import joblib at the beginning

df = pd.read_csv("dataset/Symptom2Disease.csv")
print(df.head())

print("\nChecking for missing values:\n")
print(df.isnull().sum())

df = df.drop(columns=['Unnamed: 0'])
print("\nUpdated dataset after dropping unnecessary columns:\n")
print(df.head())

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])  

X = df['text']  # Symptoms (text data)
y = df['label']  # Disease labels (numerical)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nDataset Shape:")
print("Training data:", X_train.shape, y_train.shape)
print("Testing data:", X_test.shape, y_test.shape)

model = LogisticRegression()
model.fit(X_train, y_train)

# ✅ Save the trained model and vectorizer at this step
joblib.dump(model, "models/disease_prediction_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
print("\nModel and vectorizer saved successfully!")
# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "models/disease_prediction_model.pkl")

# Save the vectorizer (TF-IDF)
joblib.dump(vectorizer, "models/vectorizer.pkl")

# Save the Label Encoder (INSERT THIS HERE)
joblib.dump(label_encoder, "models/label_encoder.pkl")

print("\nModel and vectorizer saved successfully!")
print("\nLabel encoder saved successfully!")  # ✅ Confirms that label encoder is saved.


# ✅ Reload the saved model and vectorizer
model = joblib.load("models/disease_prediction_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

def predict_disease(symptoms):
    input_data = vectorizer.transform([symptoms])  
    prediction = model.predict(input_data)
    predicted_label = label_encoder.inverse_transform(prediction)[0]  
    return predicted_label

user_symptoms = input("\nEnter your symptoms: ")
predicted_disease = predict_disease(user_symptoms)

print(f"\nPredicted Disease: {predicted_disease}")
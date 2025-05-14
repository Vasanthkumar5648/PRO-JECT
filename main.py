import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv(r"D:\download\Telco_Customer_Churn (1).csv")

data.drop(columns=['customerID'], inplace=True)

le = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = le.fit_transform(data[column])

data.fillna(data.mean(), inplace=True)

X = data.drop(columns=['Churn'])
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

import joblib
joblib.dump(model, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')


def predict_churn(input_data):
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    input_data = scaler.transform([input_data])
    prediction = model.predict(input_data)
    return "Churn" if prediction[0] == 1 else "Not Churn"
# Example Usage
sample_data = X_test[0]
print("Prediction for sample customer:", predict_churn(sample_data))
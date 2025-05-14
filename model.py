# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import joblib
import warnings

warnings.filterwarnings('ignore')


# Load and preprocess data
def load_and_preprocess():
    data = pd.read_csv(r"D:\download\Telco_Customer_Churn (1).csv")

    # EDA: Check missing values and data types
    print("Missing values:\n", data.isnull().sum())
    print("\nData types:\n", data.dtypes)

    # Drop customer ID and visualize churn distribution
    data.drop(columns=['customerID'], inplace=True)
    sns.countplot(x='Churn', data=data)
    plt.title('Churn Distribution')
    plt.show()

    # Convert categorical features
    le = LabelEncoder()
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = le.fit_transform(data[column])

    # Handle missing values with imputation
    imputer = SimpleImputer(strategy='median')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    return data


# Train model
def train_model(data):
    X = data.drop(columns=['Churn'])
    y = data['Churn']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Random Forest with hyperparameters
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.show()

    return model, scaler


# Save artifacts
def save_artifacts(model, scaler):
    joblib.dump(model, 'churn_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model and scaler saved successfully")


# Prediction function
def predict_churn(input_data):
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    input_data = scaler.transform([input_data])
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0]
    return {
        "prediction": "Churn" if prediction[0] == 1 else "Not Churn",
        "confidence": float(np.max(probability)),
        "churn_probability": float(probability[1])
    }


# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    data = load_and_preprocess()

    # Train and evaluate model
    model, scaler = train_model(data)

    # Save artifacts
    save_artifacts(model, scaler)

    # Example prediction
    sample_data = data.drop(columns=['Churn']).iloc[0]
    result = predict_churn(sample_data)
    print(f"\nSample Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Churn Probability: {result['churn_probability']:.2%}")
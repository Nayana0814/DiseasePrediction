import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier  # or whatever model you used
import pickle
import os

# ---- Set MLflow Experiment ----
mlflow.set_experiment("Disease Prediction")

# ================================================
# 1. DIABETES MODEL
# ================================================
with mlflow.start_run(run_name="Diabetes Model"):

    # Load data
    df = pd.read_csv(r'C:\Users\NAYANA\Desktop\DiseasePrediction\Datasets\diabetes.csv')
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Log parameters
    mlflow.log_param("test_size", 0.3)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("model_type", "RandomForestClassifier")

    # Train
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("accuracy",  accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall",    recall)
    mlflow.log_metric("f1_score",  f1)

    # Log model to MLflow
    mlflow.sklearn.log_model(model, "diabetes_model")

    # Save .sav file as before (so app.py still works)
    pickle.dump(model, open('diabetes_model.sav', 'wb'))

    print(f"Diabetes - Accuracy: {accuracy:.2f}, F1: {f1:.2f}")


# ================================================
# 2. HEART DISEASE MODEL
# ================================================
with mlflow.start_run(run_name="Heart Disease Model"):

    df = pd.read_csv(r'C:\Users\NAYANA\Desktop\DiseasePrediction\Datasets\heart.csv')
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("model_type", "RandomForestClassifier")

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy",  accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall",    recall)
    mlflow.log_metric("f1_score",  f1)

    mlflow.sklearn.log_model(model, "heart_model")
    pickle.dump(model, open('heart_model.sav', 'wb'))

    print(f"Heart - Accuracy: {accuracy:.2f}, F1: {f1:.2f}")


# ================================================
# 3. PARKINSON'S MODEL
# ================================================
with mlflow.start_run(run_name="Parkinsons Model"):

    df = pd.read_csv(r'C:\Users\NAYANA\Desktop\DiseasePrediction\Datasets\parkinsons.csv')
    df = df.drop('name', axis=1)
    X = df.drop('status', axis=1)
    y = df['status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("model_type", "RandomForestClassifier")

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy",  accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall",    recall)
    mlflow.log_metric("f1_score",  f1)

    mlflow.sklearn.log_model(model, "parkinsons_model")
    pickle.dump(model, open('parkinsons_model.sav', 'wb'))

    print(f"Parkinsons - Accuracy: {accuracy:.2f}, F1: {f1:.2f}")
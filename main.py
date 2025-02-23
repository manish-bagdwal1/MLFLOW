import os
import mlflow
import mlflow.sklearn
import dvc.api
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

import dvc.api
import pandas as pd
import fsspec

# Dataset path in DVC
DATA_PATH = "data/Customer_Churn_Dataset.csv"

# Get URL from DVC
dvc_url = dvc.api.get_url(DATA_PATH, remote="myremote", repo=".")

print(f"🔗 DVC Dataset URL: {dvc_url}")  # Debugging

# Correct the Azure path by replacing 'azure://' with 'az://'
azure_blob_url = dvc_url.replace("azure://", "az://")

# Read CSV from Azure Blob Storage
df = pd.read_csv(
    azure_blob_url,
    storage_options={"account_name": "mydvcstorage", "account_key": "DGkoaCMxRUQ/NzEjAgRCvAWPYFDErSNI28SFs3sZnYbLFAqrUerxBYr9vuImILdBrEMpxEnwYR0R+AStzYZDqA=="}
)

print("✅ Dataset loaded successfully from Azure!")


# Step 2: Preprocess Data
def preprocess_data(df):
    df.dropna(inplace=True)
    X = df.drop(columns=['churn'])  # Assuming 'target' is the label column
    y = df['churn']
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = preprocess_data(df)

# Preprocess Data

def preprocess_data(df):
    df.drop(columns=["customer_id"], inplace=True)  # Drop ID column
    df.dropna(inplace=True)
    X = df.drop(columns=['churn'])  # 'churn' is the target variable
    y = df['churn']
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = preprocess_data(df)

# Set up MLflow Tracking
mlflow.set_experiment("customer-churn-prediction")

# Train Model with Hyperparameter Tuning
def train_model(X_train, y_train, X_test, y_test):
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    model = RandomForestClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
    
    with mlflow.start_run():
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        # Predictions
        y_pred = best_model.predict(X_test)
        
        # Metrics Calculation
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log Parameters & Metrics in MLflow
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Save Model and Artifacts Separately
        model_path = "models/churn_model.pkl"
        pickle.dump(best_model, open(model_path, 'wb'))
        mlflow.sklearn.log_model(best_model, "churn_model")
        mlflow.log_artifact(model_path, artifact_path="artifacts")
        
         # ✅ Register Model in MLflow Model Registry
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/churn_model"
        mlflow.register_model(model_uri, "CustomerChurnModel")

        print("Best Model Parameters:", grid_search.best_params_)
        print(f"Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

# Run Training
train_model(X_train, y_train, X_test, y_test)



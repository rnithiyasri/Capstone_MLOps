import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# -----------------------------
# MLflow Configuration
# -----------------------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("BreastCancerPrediction")


# -----------------------------
# Load DVC-tracked dataset
# -----------------------------
df = pd.read_csv("data/breast_cancer.csv")

X = df.drop("target", axis=1)
y = df["target"]


# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# Models
# -----------------------------
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=10000))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


# -----------------------------
# Train and Compare Models
# -----------------------------
best_accuracy = 0
best_model = None
best_model_name = None

os.makedirs("models", exist_ok=True)

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        print(f"{name}: {accuracy:.4f}")

        # MLflow logging
        mlflow.log_param("model", name)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
            model,
            "model"
        )

        # Select best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name


# -----------------------------
# Save Best Model
# -----------------------------
joblib.dump(
    best_model,
    "models/best_model.pkl"
)

print("\nBest Model:", best_model_name)
print("Best Accuracy:", best_accuracy)
print("Best model saved successfully.")
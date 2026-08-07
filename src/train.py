from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import mlflow
import mlflow.sklearn
import joblib
import os
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("BreastCancerPrediction")
# Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

best_accuracy = 0
best_model = None

os.makedirs("models", exist_ok=True)

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        print(f"{name}: {accuracy:.4f}")

        mlflow.log_param("model", name)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(model, "model")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

joblib.dump(best_model, "models/best_model.pkl")

print("\nBest Accuracy:", best_accuracy)
print("Best model saved successfully.")
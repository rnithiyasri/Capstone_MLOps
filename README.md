# Capstone MLOps Project

## Overview

This project demonstrates an end-to-end MLOps workflow for a machine learning application using the Breast Cancer dataset.

The project includes data versioning with DVC, experiment tracking and model registration with MLflow, a FastAPI prediction service, Docker containerization, and CI/CD automation using GitHub Actions.

## MLOps Workflow

```text
Dataset
   ↓
DVC Data Versioning
   ↓
Data Preparation
   ↓
Train Multiple ML Models
   ↓
MLflow Experiment Tracking
   ↓
Compare Model Performance
   ↓
Register Best Model
   ↓
FastAPI Prediction API
   ↓
Docker
   ↓
GitHub Actions CI/CD
Machine Learning Models

Three classification models were trained and compared:

Random Forest
Decision Tree
Logistic Regression
Model Performance
Model	Accuracy
Random Forest	0.956
Decision Tree	0.912
Logistic Regression	0.982

Logistic Regression achieved the highest accuracy of 0.982 and was registered as the best-performing model.

MLflow

MLflow is used for:

Experiment tracking
Logging parameters
Logging metrics
Storing model artifacts
Model registration
Registered Model
Model Name: BreastCancerModel
Version: 2
Model: Logistic Regression
Accuracy: 0.982
DVC

DVC (Data Version Control) is used to track and version the dataset and ML pipeline.

DVC status was successfully verified:

Data and pipelines are up to date.

Important DVC components include:

.dvc/
.dvcignore
data/
FastAPI Prediction API

A FastAPI application is used to provide predictions through a REST API.

Prediction Endpoint
POST /predict
Run the API
uvicorn src.app:app --reload

The API will run at:

http://127.0.0.1:8000

Swagger API documentation is available at:

http://127.0.0.1:8000/docs

The /predict endpoint accepts the required input features and returns the predicted class.

Example response:

{
  "prediction": 1
}
Docker

The application is containerized using Docker.

Build the Docker image using:

docker build -t capstone-mlops .
GitHub Actions CI/CD

GitHub Actions is used to automate the project workflow.

The CI/CD pipeline performs the following tasks:

Checkout the repository
Set up Python
Install dependencies
Run tests
Prepare the data
Train the model
Evaluate the model
Build and deploy the application

The workflow is triggered by changes pushed to the main branch.

Run the Project Locally
Clone the Repository
git clone https://github.com/rnithiyasri/Capstone_MLOps.git
cd Capstone_MLOps
Install Dependencies
pip install -r requirements.txt
Run the Training
python src/train.py
Start FastAPI
uvicorn src.app:app --reload
Open Swagger
http://127.0.0.1:8000/docs
Project Structure
Capstone_MLOps/
│
├── .dvc/
├── .github/
│   └── workflows/
│
├── data/
├── models/
├── mlartifacts/
│
├── src/
│   ├── app.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
├── tests/
├── Dockerfile
├── requirements.txt
├── mlflow.db
├── .dvcignore
└── README.md
Technologies Used
Python
Scikit-learn
MLflow
DVC
FastAPI
Docker
Git
GitHub
GitHub Actions

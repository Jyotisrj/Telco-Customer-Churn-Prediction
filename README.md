# Telco Customer Churn Prediction

A machine learning project to predict whether a telecom customer is likely to churn based on customer demographics, services, contract details, and billing-related information.

## 📌 Project Overview

Customer churn is an important business problem for telecom companies. This project analyzes customer data, performs data preprocessing and exploratory data analysis, and compares multiple machine learning classification models for predicting customer churn.

## 🎯 Objectives

* Analyze customer churn patterns through Exploratory Data Analysis (EDA)
* Clean and transform the dataset
* Handle categorical variables using one-hot encoding
* Perform feature scaling and feature selection
* Train multiple classification models
* Evaluate and compare model performance

## 📊 Dataset

The project uses the **Telco Customer Churn** dataset containing:

* **7,043 records**
* **21 columns**
* Target variable: `Churn`

The dataset contains information related to customer demographics, tenure, telecom services, contracts, payment methods, and billing.

## 🔄 Project Workflow

1. Import required Python libraries
2. Load the dataset
3. Perform Exploratory Data Analysis (EDA)
4. Detect outliers using the IQR method
5. Clean and transform the data
6. Apply one-hot encoding
7. Rearrange features
8. Perform feature scaling
9. Perform feature selection
10. Train classification models
11. Evaluate model performance using accuracy, classification reports, and confusion matrices

## 🤖 Machine Learning Models

The following models were implemented and compared:

* Logistic Regression
* Support Vector Classifier (SVC)
* Decision Tree Classifier
* K-Nearest Neighbors (KNN)

## 📈 Model Performance

| Model                     | Test Accuracy |
| ------------------------- | ------------: |
| Logistic Regression       |        80.03% |
| Support Vector Classifier |        80.12% |
| Decision Tree Classifier  |        73.21% |
| KNN Classifier            |        79.22% |


| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score (Churn) |
| :-------------------------------- | :------- | :---------------- | :------------- | :--------------- |
| Logistic Regression (Original) | 0.80 | 0.65 | 0.53 | 0.58 |
| SVC | 0.80 | 0.67 | 0.48 | 0.56 |
| Decision Tree | 0.73 | 0.49 | 0.51 | 0.50 |
| KNN | 0.79 | 0.62 | 0.55 | 0.58 |

The notebook also generates classification reports and confusion matrices for evaluating the predictions.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook / Google Colab

## 📂 Project Structure

```text
Telco-Customer-Churn-Prediction/
│
├── Customer_Churn_Prediction.ipynb
├── README.md
├── requirements.txt
├── screenshots/
│   ├── churn_distribution.png
│   ├── correlation_heatmap.png
│   ├── model_comparison.png
│   └── confusion_matrix.png
└── .gitignore
```

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Jyotisrj/Telco-Customer-Churn-Prediction.git
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the notebook

Open:

```text
Customer_Churn_Prediction.ipynb
```

You can run it using **Jupyter Notebook** or **Google Colab**.

### 4. Dataset Path

Update the dataset path in the notebook according to the location of your CSV file before running the project.

## 📌 Key Outcomes

* Performed complete data preprocessing and exploratory analysis on 7,043 customer records.
* Compared four machine learning classification algorithms.
* Achieved approximately **80% test accuracy** with Logistic Regression and SVC.
* Used classification reports and confusion matrices to analyze model predictions.
* Built an end-to-end customer churn prediction workflow using Python and Scikit-learn.

## 🚀 Future Improvements

* Hyperparameter tuning using GridSearchCV
* Try additional boosting models such as AdaBoost, Gradient Boosting, CatBoost, and XGBoost
* Improve recall for the churn class
* Deploy the prediction model using Streamlit
* Build an interactive customer churn dashboard

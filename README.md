# Telco-Customer-Churn-Prediction

## Project Overview

This project focuses on predicting customer churn for a telecommunications company using machine learning techniques. Customer churn is a critical business problem as it directly impacts revenue and growth. By identifying customers at high risk of churning, the company can implement targeted retention strategies, which are often more cost-effective than acquiring new customers.

## Dataset

The dataset used for this analysis is the [Telco Customer Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn) from Kaggle. It contains information about customers' demographics, services they have subscribed to, contract details, and whether they have churned or not.

### Data Fields:

*   `customerID`: Customer ID
*   `gender`: Whether the customer is a male or a female
*   `SeniorCitizen`: Whether the customer is a senior citizen or not (1, 0)
*   `Partner`: Whether the customer has a partner or not (Yes, No)
*   `Dependents`: Whether the customer has dependents or not (Yes, No)
*   `tenure`: Number of months the customer has stayed with the company
*   `PhoneService`: Whether the customer has phone service or not (Yes, No)
*   `MultipleLines`: Whether the customer has multiple lines or not (Yes, No, No phone service)
*   `InternetService`: Customer’s internet service provider (DSL, Fiber optic, No)
*   `OnlineSecurity`: Whether the customer has online security or not (Yes, No, No internet service)
*   `OnlineBackup`: Whether the customer has online backup or not (Yes, No, No internet service)
*   `DeviceProtection`: Whether the customer has device protection or not (Yes, No, No internet service)
*   `TechSupport`: Whether the customer has tech support or not (Yes, No, No internet service)
*   `StreamingTV`: Whether the customer has streaming TV or not (Yes, No, No internet service)
*   `StreamingMovies`: Whether the customer has streaming movies or not (Yes, No, No internet service)
*   `Contract`: The contract term of the customer (Month-to-month, One year, Two year)
*   `PaperlessBilling`: Whether the customer has paperless billing or not (Yes, No)
*   `PaymentMethod`: The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
*   `MonthlyCharges`: The amount charged to the customer monthly
*   `TotalCharges`: The total amount charged to the customer
*   `Churn`: Whether the customer churned or not (Yes or No) - **Target Variable**

## Key Steps and Methodology

1.  **Data Loading & Initial Exploration:** Loaded the dataset using Pandas and performed initial checks for shape, data types, missing values, and duplicates.
2.  **Data Cleaning & Preprocessing:**
    *   Converted `TotalCharges` to numeric type, handling `NaN` values (imputed with mean).
    *   Dropped the `customerID` column as it's not relevant for modeling.
    *   Applied One-Hot Encoding to categorical features to convert them into a numerical format suitable for machine learning algorithms.
    *   Rearranged columns for better organization.
    *   Applied Feature Scaling using `StandardScaler` to normalize numerical features.
3.  **Exploratory Data Analysis (EDA):**
    *   Analyzed distributions of numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) and categorical features.
    *   Examined feature distributions in relation to the `Churn` target variable.
    *   Identified that the dataset is imbalanced, with significantly more 'No Churn' instances than 'Yes Churn'.
4.  **Addressing Data Imbalance:** Utilized the Synthetic Minority Over-sampling Technique (SMOTE) to create synthetic samples for the minority class ('Churn_Yes') in the training data, helping to balance the dataset.
5.  **Model Training & Evaluation:** Trained and evaluated several classification models:
    *   Logistic Regression (Original)
    *   Support Vector Classifier (SVC)
    *   Decision Tree Classifier
    *   K-Nearest Neighbors (KNN)
    *   Logistic Regression (with SMOTE-resampled data)
    *   Logistic Regression (Tuned with GridSearchCV on SMOTE-resampled data)
6.  **Hyperparameter Tuning:** Performed `GridSearchCV` on the Logistic Regression model to find optimal hyperparameters (e.g., `C` and `solver`) for improved performance.
7.  **Feature Importance Analysis:** Examined feature coefficients from the tuned Logistic Regression model to identify the most influential factors contributing to churn.
8.  **Model Performance Comparison:** Compared the performance of all trained models using relevant metrics such as accuracy, precision, recall, and F1-score, with a particular focus on the 'Churn' class due to the dataset's imbalance.

## Key Findings (To be filled in after full execution)

*   **EDA Insights:** 
    *   Customers on month-to-month contracts have a significantly higher churn rate compared to those on one-year or two-year contracts.
    *   Customers with lower `tenure` and `TotalCharges` show a higher propensity to churn.
    *   (Add more specific insights here from your EDA plots and analyses)
*   **Feature Importance:**
    *   The most influential features in predicting churn (based on tuned Logistic Regression) were found to be: `Contract_Month-to-month`, `InternetService_Fiber optic`, `tenure`, and `MonthlyCharges`. (Confirm with your feature importance plot and potentially add more details).
*   **Best Performing Model:**
    *   Based on the F1-score for the churn class, the **Logistic Regression model trained with SMOTE and further optimized through GridSearchCV** performed best. (Confirm this with your final comparison table).

## Model Performance Summary (To be filled in after full execution)

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score (Churn) |
| :-------------------------------- | :------- | :---------------- | :------------- | :--------------- |
| Logistic Regression (Original) | `0.80  ` | ` 0.65` | `0.53` | `0.58` |
| SVC | `0.80` | `0.67` | `0.48` | `0.56` |
| Decision Tree | `0.73` | `0.49` | `0.51` | `0.50` |
| KNN | `0.79` | `0.62` | `0.55` | `0.58` |


## Business Recommendations

Based on the analysis and model performance, here are some actionable business recommendations to reduce customer churn:

1.  **Focus on Month-to-Month Contract Customers:** Implement targeted retention campaigns (e.g., offering discounts for longer-term commitments, personalized bundle offers) specifically for this segment.
2.  **Monitor Early-Tenure Customers:** Enhance onboarding processes, provide dedicated support for the first few months, and offer incentives to encourage engagement during this critical period.
3.  **Address High Monthly Charges:** Ensure pricing is competitive and transparent. Value-added services or loyalty programs could help justify higher costs for customers.
4.  **Proactive Interventions:** Use the best-performing predictive model to identify high-risk customers *before* they churn. Customer service teams can then reach out with personalized offers or support to prevent attrition.
5.  **Continuous Monitoring:** Churn drivers can change over time. Regularly re-evaluate the model's performance and retrain it with new data to ensure its effectiveness and adapt to evolving customer behavior.

## Technologies Used

*   Python
*   Pandas (for data manipulation)
*   NumPy (for numerical operations)
*   Matplotlib (for visualizations)
*   Seaborn (for visualizations)
*   Scikit-learn (for machine learning models, preprocessing, and evaluation)
*   Imbalanced-learn (for handling imbalanced datasets like SMOTE)


## Future Enhancements

*   Explore other advanced classification algorithms (e.g., XGBoost, LightGBM).
*   Implement more sophisticated feature engineering techniques.
*   Conduct A/B testing on retention strategies derived from the model's insights.
*   Deploy the model into a production environment for real-time churn prediction.

## Contact

If you have any questions or feedback, feel free to reach out:

*   **Jyoti**

 Credit Risk Prediction — Loan Default Classification
🧠 Task Objective

The objective of this project is to build a machine learning model that can predict whether a loan application will be approved or rejected based on applicant data. This helps financial institutions assess credit risk and make informed lending decisions.

The model classifies applicants into:

Approved (Low Risk)
Default / Rejected (High Risk)
⚙️ Approach
1. Data Handling
Attempted to load the Loan Prediction dataset
Generated a synthetic dataset if the original dataset is unavailable (ensures code always runs)
2. Data Preprocessing
Handled missing values:
Categorical → filled with mode
Numerical → filled with median
Removed irrelevant column (Loan_ID)
3. Feature Engineering

Created new meaningful features:

TotalIncome = Applicant + Co-applicant income
LoanToIncome Ratio
EMI (Estimated Monthly Installment)
Balance Income
4. Encoding & Scaling
Applied Label Encoding to categorical variables
Used StandardScaler for feature scaling
5. Exploratory Data Analysis (EDA)

Visualizations include:

Loan amount distribution
Income distribution
Education vs loan status
Credit history impact
Property area trends
Correlation heatmap
6. Model Building

Two models were trained and compared:

Logistic Regression
Decision Tree Classifier
7. Evaluation Metrics
Accuracy Score
Confusion Matrix
Classification Report
📈 Results and Insights
🏆 Best Performing Model
The best model is selected based on highest accuracy
🔍 Key Insights
Credit History is the most important factor in predicting loan approval
Applicants with good credit history are far more likely to be approved
Income and loan amount significantly influence decisions
Decision Tree captures complex patterns better
Logistic Regression provides better interpretability
📊 Visual Outputs

The project generates:

EDA plots → credit_risk_eda.png
Confusion matrices → credit_risk_confusion_matrices.png
Feature importance chart → credit_risk_feature_importance.png
Accuracy comparison → credit_risk_accuracy_comparison.png
🚀 Conclusion

This project demonstrates how machine learning can effectively automate credit risk assessment. By combining data preprocessing, feature engineering, and model evaluation, we can build reliable systems to support financial decision-making

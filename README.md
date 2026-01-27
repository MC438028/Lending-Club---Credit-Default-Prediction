# Lending-Club---Credit-Default-Prediction
## Project Overview
This project focuses on predicting loan default risk using historical lending data from Lending Club.
By leveraging borrower demographics, credit history, and loan characteristics, the project aims to build robust classification models that can accurately identify high-risk applicants and support data-driven credit decision-making.

Unlike conventional approaches that rely on a single feature selection technique, this project proposes a weighted integrated feature selection framework, which combines multiple feature selection methods to improve model robustness, generalization ability, and risk–return balance.

## Key Objectives
*Predict whether a borrower will default on a loan using supervised machine learning models
*Address challenges such as high-dimensional features, feature redundancy, and class imbalance
*Compare single-method feature selection with a weighted integrated feature selection strategy
*Evaluate model performance using metrics relevant to credit risk (ROC-AUC, Recall, FPR, RMSE)

## Methodology

**Data Source:** Lending Club dataset
**Feature Selection Methods:**
*Filter methods (Mutual Information, ANOVA)
*Wrapper methods (RFE)
*Embedded methods (L1 regularization, Random Forest, XGBoost, LightGBM importance)
**Models Used:** LightGBM、XGBoost、Random Forest、LR、ANN、Hybrid MOdel

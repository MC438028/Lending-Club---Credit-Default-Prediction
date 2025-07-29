import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve, auc,
                             accuracy_score, mean_squared_error)
import xgboost as xgb
import lightgbm as lgb
import os
import time


train_data= pd.read_csv('/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/dataset/train_data_encoded.csv')
test_data = pd.read_csv('/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/dataset/test_data_encoded.csv')
X_train = train_data.drop(columns=['loan_status_flag'])
y_train = train_data['loan_status_flag']
X_test = test_data.drop(columns=['loan_status_flag'])
y_test = test_data['loan_status_flag']

# Configuration
plt.rcParams["font.family"] = ["Arial", "sans-serif"]
sns.set(font_scale=1.2)
os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/text", exist_ok=True)

# Define feature subset
feature_subset = [
    'fico_range_low', 'acc_open_past_24mths', 'sub_grade', 'term', 'inq_last_6mths', 
    'int_rate', 'installment', 'mort_acc', 'percent_bc_gt_75', 'mths_since_recent_inq', 
    'mo_sin_rcnt_tl', 'verification_status_Source Verified', 'verification_status_Not Verified', 
    'open_act_il', 'verification_status_Verified', 'inq_last_12m', 'total_bc_limit', 'dti', 
    'all_util', 'funded_amnt', 'mo_sin_old_rev_tl_op', 'tot_hi_cred_lim', 'avg_cur_bal', 
    'disbursement_method', 'mths_since_rcnt_il', 'loan_amnt', 'annual_inc', 'mths_since_recent_bc', 
    'home_ownership_ANY', 'num_rev_tl_bal_gt_0', 'home_ownership_MORTGAGE', 'home_ownership_OWN', 
    'home_ownership_NONE', 'home_ownership_OTHER', 'home_ownership_RENT', 'emp_title_cluster', 
    'earliest_cr_line_year', 'mo_sin_old_il_acct', 'emp_length'
]

# Initialize results storage
all_results = {}
all_reports = {}
# Feature Selector
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
        
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            missing_features = [f for f in self.feature_names if f not in X.columns]
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
        return self
    
    def transform(self, X):
        return X[self.feature_names]

# Visualization functions
def plot_confusion_matrix(cm, model_name, class_names):
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'results/plots/{model_name}_confusion_matrix.png', dpi=300)
    plt.close()

def plot_roc_curve(y_true, y_score, title):
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{title} Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'results/plots/{title}_roc_curve.png', dpi=300)
    plt.close()

# Result saving functions
def save_model_report(model_name, result, report):
    # Save classification report
    with open(f'results/text/{model_name}_classification_report.txt', 'w') as f:
        f.write(f"{model_name} Classification Report:\n")
        f.write("-" * 50 + "\n")
        f.write(report)
    
    # Save model results
    with open(f'results/text/{model_name}_results.txt', 'w') as f:
        f.write(f"{model_name} Results:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Best Parameters: {result['best_params']}\n")
        f.write(f"Training Time: {result['training_time']:.2f} seconds\n")
        f.write(f"Training ROC-AUC: {result['train_roc_auc']:.4f}\n")
        f.write(f"Testing ROC-AUC: {result['test_roc_auc']:.4f}\n")
        f.write(f"Accuracy: {result['accuracy']:.4f}\n")
        f.write(f"RMSE: {result['rmse']:.4f}\n")
        f.write(f"Recall(Default): {result['tpr']:.4f}\n")
        f.write(f"FPR: {result['fpr']:.4f}\n\n")
        
        # Confusion matrix details
        cm = result['confusion_matrix']
        f.write("Confusion Matrix:\n")
        f.write(f"  True Negatives: {cm[0, 0]}\n")
        f.write(f"  False Positives: {cm[0, 1]}\n")
        f.write(f"  False Negatives: {cm[1, 0]}\n")
        f.write(f"  True Positives: {cm[1, 1]}\n")


from sklearn.ensemble import RandomForestClassifier

def train_random_forest(feature_subset, X_train, X_test, y_train, y_test):
    model_name = "RandomForest"
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Define model and parameters
    model = RandomForestClassifier(
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    params = {
        'model__n_estimators': [500],
        'model__max_features': ['sqrt'],
        'model__max_depth': [15]
    }
    
    # Create pipeline
    pipeline = Pipeline([
        ('feature_selector', FeatureSelector(feature_subset)),
        ('model', model)
    ])
    
    # Grid search
    gs = GridSearchCV(
        pipeline, 
        param_grid=params,
        scoring='roc_auc',
        cv=stratified_cv,
        n_jobs=-1,
        verbose=1
    )
    
    # Train
    start_time = time.time()
    gs.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Evaluate
    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    train_roc_auc = roc_auc_score(y_train, best_model.predict_proba(X_train)[:, 1])
    test_roc_auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_proba))
    report = classification_report(y_test, y_pred)
    
    # Store results
    result = {
        'best_params': gs.best_params_,
        'train_roc_auc': train_roc_auc,
        'test_roc_auc': test_roc_auc,
        'accuracy': accuracy,
        'rmse': rmse,
        'tpr': tpr,
        'fpr': fpr,
        'y_pred_proba': y_pred_proba,
        'training_time': training_time,
        'confusion_matrix': cm
    }
    
    # Visualize and save
    plot_confusion_matrix(cm, model_name, ['Non-default', 'Default'])
    plot_roc_curve(y_test, y_pred_proba, model_name)
    save_model_report(model_name, result, report)
    
    # Print summary
    print(f"\n{model_name} Results:")
    print(f"Best Parameters: {gs.best_params_}")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Training ROC-AUC: {train_roc_auc:.4f}")
    print(f"Test ROC-AUC: {test_roc_auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Recall for Default Class (TPR): {tpr:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    
    return result, report

# Run training
rf_result, rf_report = train_random_forest(feature_subset, X_train, X_test, y_train, y_test)
all_results['RandomForest'] = rf_result
all_reports['RandomForest'] = rf_report

# Save checkpoint
with open('results/rf_results.pkl', 'wb') as f:
    pickle.dump(rf_result, f)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def train_linear_regression(feature_subset, X_train, X_test, y_train, y_test):
    model_name = "Linear Regression"
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Define model and parameters
    model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    params = {
        'model__C': [100],
        'model__tol': [1e-3, 1e-4, 1e-5]
    }
    
    # Create pipeline
    pipeline = Pipeline([
        ('feature_selector', FeatureSelector(feature_subset)),
        ('scaler', StandardScaler()),  
        ('model', model)
    ])
    
    # Grid search
    gs = GridSearchCV(
        pipeline, 
        param_grid=params,
        scoring='roc_auc',
        cv=stratified_cv,
        n_jobs=-1,
        verbose=1
    )
    
    
    # Train
    start_time = time.time()
    gs.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Evaluate
    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    train_roc_auc = roc_auc_score(y_train, best_model.predict_proba(X_train)[:, 1])
    test_roc_auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_proba))
    report = classification_report(y_test, y_pred)
    
    # Store results
    result = {
        'best_params': gs.best_params_,
        'train_roc_auc': train_roc_auc,
        'test_roc_auc': test_roc_auc,
        'accuracy': accuracy,
        'rmse': rmse,
        'tpr': tpr,
        'fpr': fpr,
        'y_pred_proba': y_pred_proba,
        'training_time': training_time,
        'confusion_matrix': cm
    }
    
    # Visualize and save
    plot_confusion_matrix(cm, model_name, ['Non-default', 'Default'])
    plot_roc_curve(y_test, y_pred_proba, model_name)
    save_model_report(model_name, result, report)
    
    # Print summary
    print(f"\n{model_name} Results:")
    print(f"Best Parameters: {gs.best_params_}")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Training ROC-AUC: {train_roc_auc:.4f}")
    print(f"Test ROC-AUC: {test_roc_auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Recall for Default Class (TPR): {tpr:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    
    return result, report

# Run training
linear_regression_result, linear_regression_report = train_linear_regression(feature_subset, X_train, X_test, y_train, y_test)
all_results['Linear Regression'] = linear_regression_result
all_reports['Linear Regression'] = linear_regression_report

# Save checkpoint
with open('results/linear_regression_results.pkl', 'wb') as f:
    pickle.dump(linear_regression_result, f)

import xgboost as xgb


def train_xgboost(feature_subset, X_train, X_test, y_train, y_test):
    model_name = "XGBoost"
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
       
    # Define model and parameters
    model = xgb.XGBClassifier(
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
        random_state=42, n_jobs=-1,
        eval_metric='auc'
    )
    params = {
        'model__n_estimators': [500],
        'model__max_depth': [6],
        'model__learning_rate': [0.1]
    }
    
    # Create pipeline
    pipeline = Pipeline([
        ('feature_selector', FeatureSelector(feature_subset)),
        ('model', model)
    ])
    
    # Grid search
    gs = GridSearchCV(
        pipeline, 
        param_grid=params,
        scoring='roc_auc',
        cv=stratified_cv,
        n_jobs=-1,
        verbose=1
    )
    
    # Train
    start_time = time.time()
    gs.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Evaluate
    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    train_roc_auc = roc_auc_score(y_train, best_model.predict_proba(X_train)[:, 1])
    test_roc_auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_proba))
    report = classification_report(y_test, y_pred)
    
    # Store results
    result = {
        'best_params': gs.best_params_,
        'train_roc_auc': train_roc_auc,
        'test_roc_auc': test_roc_auc,
        'accuracy': accuracy,
        'rmse': rmse,
        'tpr': tpr,
        'fpr': fpr,
        'y_pred_proba': y_pred_proba,
        'training_time': training_time,
        'confusion_matrix': cm
    }
    
    # Visualize and save
    plot_confusion_matrix(cm, model_name, ['Non-default', 'Default'])
    plot_roc_curve(y_test, y_pred_proba, model_name)
    save_model_report(model_name, result, report)
    
    # Print summary
    print(f"\n{model_name} Results:")
    print(f"Best Parameters: {gs.best_params_}")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Training ROC-AUC: {train_roc_auc:.4f}")
    print(f"Test ROC-AUC: {test_roc_auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Recall for Default Class (TPR): {tpr:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    
    return result, report

# Run training
xgb_result, xgb_report = train_xgboost(feature_subset, X_train, X_test, y_train, y_test)
all_results['XGBoost'] = xgb_result
all_reports['XGBoost'] = xgb_report

# Save checkpoint
with open('results/xgb_results.pkl', 'wb') as f:
    pickle.dump(xgb_result, f)
import lightgbm as lgb

def train_lightgbm(feature_subset, X_train, X_test, y_train, y_test):
    model_name = "LightGBM"
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Define model and parameters
    model = lgb.LGBMClassifier(
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    params = {
        'model__n_estimators': [500],
        'model__max_depth': [6],
        'model__learning_rate': [0.1]
    }
    
    # Create pipeline
    pipeline = Pipeline([
        ('feature_selector', FeatureSelector(feature_subset)),
        ('model', model)
    ])
    
    # Grid search
    gs = GridSearchCV(
        pipeline, 
        param_grid=params,
        scoring='roc_auc',
        cv=stratified_cv,
        n_jobs=-1,
        verbose=1
    )
    
    # Train
    start_time = time.time()
    gs.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Evaluate
    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    train_roc_auc = roc_auc_score(y_train, best_model.predict_proba(X_train)[:, 1])
    test_roc_auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_proba))
    report = classification_report(y_test, y_pred)
    
    # Store results
    result = {
        'best_params': gs.best_params_,
        'train_roc_auc': train_roc_auc,
        'test_roc_auc': test_roc_auc,
        'accuracy': accuracy,
        'rmse': rmse,
        'tpr': tpr,
        'fpr': fpr,
        'y_pred_proba': y_pred_proba,
        'training_time': training_time,
        'confusion_matrix': cm
    }
    
    # Visualize and save
    plot_confusion_matrix(cm, model_name, ['Non-default', 'Default'])
    plot_roc_curve(y_test, y_pred_proba, model_name)
    save_model_report(model_name, result, report)
    
    # Print summary
    print(f"\n{model_name} Results:")
    print(f"Best Parameters: {gs.best_params_}")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Training ROC-AUC: {train_roc_auc:.4f}")
    print(f"Test ROC-AUC: {test_roc_auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Recall for Default Class (TPR): {tpr:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    
    return result, report

# Run training
lgb_result, lgb_report = train_lightgbm(feature_subset, X_train, X_test, y_train, y_test)
all_results['LightGBM'] = lgb_result
all_reports['LightGBM'] = lgb_report

# Save checkpoint
with open('results/lgb_results.pkl', 'wb') as f:
    pickle.dump(lgb_result, f)

%pip install imbalanced-learn
from sklearn.preprocessing import StandardScaler 
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE  # 用于过采样少数类
from imblearn.pipeline import Pipeline as ImbPipeline  # 支持采样的管道
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, mean_squared_error, classification_report
import time
import pickle

def train_ann(feature_subset, X_train, X_test, y_train, y_test):
    model_name = "ANN"
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 定义模型及参数（无需样本权重，用SMOTE处理不平衡）
    model = MLPClassifier(
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        learning_rate_init=0.001,
        random_state=42
    )
    params = {
        'model__hidden_layer_sizes': [(50, 50, 50)],
        'model__activation': ['relu'],
        'model__solver': ['adam'],
        'model__max_iter': [500]
    }

    # 创建带SMOTE的管道（ImbPipeline支持采样步骤）
    pipeline = ImbPipeline([
        ('feature_selector', FeatureSelector(feature_subset)),  # 特征选择
        ('scaler', StandardScaler()),  # 标准化
        ('smote', SMOTE(random_state=42)),  # 过采样少数类（解决不平衡）
        ('model', model)  # ANN模型
    ])

    # 网格搜索
    gs = GridSearchCV(
        pipeline,
        param_grid=params,
        scoring='roc_auc',
        cv=stratified_cv,
        n_jobs=-1,
        verbose=1
    )

    # 训练：无需传递样本权重，SMOTE已处理不平衡
    start_time = time.time()
    gs.fit(X_train, y_train)  # 不再需要sample_weight参数
    training_time = time.time() - start_time

    # 评估
    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # 计算指标（避免除零错误）
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0.0
    train_roc_auc = roc_auc_score(y_train, best_model.predict_proba(X_train)[:, 1]) if len(np.unique(y_train)) > 1 else 0.5
    test_roc_auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
    accuracy = accuracy_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_proba))
    report = classification_report(y_test, y_pred, zero_division=1)

    # 存储结果
    result = {
        'best_params': gs.best_params_,
        'train_roc_auc': train_roc_auc,
        'test_roc_auc': test_roc_auc,
        'accuracy': accuracy,
        'rmse': rmse,
        'tpr': tpr,
        'fpr': fpr,
        'y_pred_proba': y_pred_proba,
        'training_time': training_time,
        'confusion_matrix': cm
    }

    # 可视化与保存（假设函数已实现）
    plot_confusion_matrix(cm, model_name, ['Non-default', 'Default'])
    plot_roc_curve(y_test, y_pred_proba, model_name)
    save_model_report(model_name, result, report)

    # 打印结果
    print(f"\n{model_name} Results:")
    print(f"Best Parameters: {gs.best_params_}")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Training ROC-AUC: {train_roc_auc:.4f}")
    print(f"Test ROC-AUC: {test_roc_auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Recall for Default Class (TPR): {tpr:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")

    return result, report

# 运行训练
ann_result, ann_report = train_ann(feature_subset, X_train, X_test, y_train, y_test)
all_results['ANN'] = ann_result
all_reports['ANN'] = ann_report

# 保存结果
with open('results/ann_results.pkl', 'wb') as f:
    pickle.dump(ann_result, f)


def visualize_model_comparison(results):
    """Visualize performance comparison among different models"""
    print("\nPerformance Summary of Models:")
    print("-" * 80)
    print(f"{'Model':<15} {'Train Time(s)':<12} {'Test ROC-AUC':<12} {'Accuracy':<10} {'RMSE':<8} {'Recall(Default)':<14} {'FPR'}")
    print("-" * 80)
    
    for model_name, result in sorted(results.items(), key=lambda x: x[1]['test_roc_auc'], reverse=True):
        print(f"{model_name:<15} {result['training_time']:<12.2f} {result['test_roc_auc']:<12.4f} "
              f"{result['accuracy']:<10.4f} {result['rmse']:<8.4f} "
              f"{result['tpr']:<14.4f} {result['fpr']:.4f}")
    print("-" * 80)
    
    # Visualize model performance
    plt.figure(figsize=(18, 6))
    
    # ROC-AUC and Accuracy comparison
    plt.subplot(1, 3, 1)
    metrics_data = pd.DataFrame({
        'Model': list(results.keys()),
        'ROC-AUC': [results[k]['test_roc_auc'] for k in results],
        'Accuracy': [results[k]['accuracy'] for k in results]
    })
    metrics_data = metrics_data.melt(id_vars='Model', var_name='Metric', value_name='Score')
    sns.barplot(x='Score', y='Model', hue='Metric', data=metrics_data)
    plt.title('ROC-AUC and Accuracy Comparison')
    plt.legend(loc='lower right')
    
    # Recall and FPR comparison
    plt.subplot(1, 3, 2)
    error_data = pd.DataFrame({
        'Model': list(results.keys()),
        'Recall(Default)': [results[k]['tpr'] for k in results],
        'FPR': [results[k]['fpr'] for k in results]
    })
    error_data = error_data.melt(id_vars='Model', var_name='Metric', value_name='Score')
    sns.barplot(x='Score', y='Model', hue='Metric', data=error_data)
    plt.title('Recall(Default) and FPR Comparison')
    plt.legend(loc='lower right')
    
    # RMSE comparison
    plt.subplot(1, 3, 3)
    rmse_data = pd.DataFrame({
        'Model': list(results.keys()),
        'RMSE': [results[k]['rmse'] for k in results]
    })
    sns.barplot(x='RMSE', y='Model', data=rmse_data)
    plt.title('RMSE Comparison')
    
    plt.tight_layout()
    plt.savefig('results/plots/model_comparison.png', dpi=300)
    plt.close()
    
    # Plot ROC curves for all models
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    
    for model_name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison for Different Models')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('results/plots/model_roc_curve_comparison.png', dpi=300)
    plt.show()

# Save results to text files
def save_results_to_txt(results):
    """Save numerical results to text file"""
    with open('results/text/model_comparison_summary.txt', 'w') as f:
        f.write("Performance Summary of Models:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model':<15} {'Train Time(s)':<12} {'Test ROC-AUC':<12} {'Accuracy':<10} "
                f"{'RMSE':<8} {'Recall(Default)':<14} {'FPR'}\n")
        f.write("-" * 80 + "\n")
        
        for model_name, result in sorted(results.items(), key=lambda x: x[1]['test_roc_auc'], reverse=True):
            f.write(f"{model_name:<15} {result['training_time']:<12.2f} {result['test_roc_auc']:<12.4f} "
                    f"{result['accuracy']:<10.4f} {result['rmse']:<8.4f} "
                    f"{result['tpr']:<14.4f} {result['fpr']:.4f}\n")
        f.write("-" * 80 + "\n")
        
        # Save detailed metrics for each model
        f.write("\nDetailed Metrics for Each Model:\n")
        for model_name, result in results.items():
            f.write(f"\n{model_name}:\n")
            f.write(f"  Best Parameters: {result['best_params']}\n")
            f.write(f"  Training Time: {result['training_time']:.2f} seconds\n")
            f.write(f"  Training ROC-AUC: {result['train_roc_auc']:.4f}\n")
            f.write(f"  Testing ROC-AUC: {result['test_roc_auc']:.4f}\n")
            f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"  RMSE: {result['rmse']:.4f}\n")
            f.write(f"  Recall for Default Class (TPR): {result['tpr']:.4f}\n")
            f.write(f"  False Positive Rate (FPR): {result['fpr']:.4f}\n")
            
            # Save confusion matrix
            cm = result['confusion_matrix']
            f.write("  Confusion Matrix:\n")
            f.write(f"    True Negatives: {cm[0, 0]}\n")
            f.write(f"    False Positives: {cm[0, 1]}\n")
            f.write(f"    False Negatives: {cm[1, 0]}\n")
            f.write(f"    True Positives: {cm[1, 1]}\n")

# Run comparison
save_results_to_txt(all_results)

# Save all reports
for model_name, report in all_reports.items():
    with open(f'results/text/{model_name}_classification_report.txt', 'w') as f:
        f.write(f"{model_name} Classification Report:\n")
        f.write("-" * 50 + "\n")
        f.write(report)

# Save all results to pickle
with open('results/all_model_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)

# Visualize comparison
visualize_model_comparison(all_results)
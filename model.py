import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score

# 导入预处理和特征工程函数
from data_preprocessing import (
    read_and_classify_data, process_date_columns, process_numerical_columns,
    process_median_columns, process_categorical_columns, process_mean_columns,
    process_mode_columns, process_emp_length
)
from feature_engineering import (
    process_binary_columns, process_categorical_columns as process_onehot_columns,
    process_grade, process_clustering_encoding # 导入特征映射函数
)


# --------------------------
# 1. 自定义转换器
# --------------------------
class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, func):
        self.func = func
        self.fitted_ = False

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        if not self.fitted_:
            raise RuntimeError("Please call fit() before transform()")
        return self.func(X.copy())

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

# --------------------------
# 2. 数据处理管道（返回原始特征用于映射检查）
# --------------------------
def create_data_pipeline(file_path, test_size=0.3, random_state=42):
    data = read_and_classify_data(file_path)
    print(f"Original data shape: {data.shape}")

    # Generate target variable
    default_status = [
        'Charged Off', 'Default', 
        'Does not meet the credit policy. Status:Charged Off', 
        'Late (31-120 days)'
    ]
    data['loan_status_flag'] = data['loan_status'].apply(
        lambda x: 1 if x in default_status else 0
    )

    # Split dataset (保留原始未处理的X_train_raw和X_test_raw)
    X_raw = data.drop(['loan_status_flag', 'loan_status'], axis=1, errors='ignore')
    y = data['loan_status_flag']
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=random_state
    )
    print(f"After split - Train set: {X_train_raw.shape}, Test set: {X_test_raw.shape}")

    # Define processing pipeline
    full_pipeline = Pipeline([
        ('date_processing', CustomTransformer(process_date_columns)),
        ('numerical_processing', CustomTransformer(process_numerical_columns)),
        ('median_processing', CustomTransformer(process_median_columns)),
        ('categorical_processing', CustomTransformer(process_categorical_columns)),
        ('mean_processing', CustomTransformer(process_mean_columns)),
        ('mode_processing', CustomTransformer(process_mode_columns)),
        ('emp_length_processing', CustomTransformer(process_emp_length)),
        ('binary_encoding', CustomTransformer(process_binary_columns)),
        ('onehot_encoding', CustomTransformer(process_onehot_columns)),
        ('grade_encoding', CustomTransformer(process_grade)),
        ('clustering_encoding', CustomTransformer(process_clustering_encoding))
    ])

    # Process data
    X_train_processed = full_pipeline.fit_transform(X_train_raw)
    X_test_processed = full_pipeline.transform(X_test_raw)

    # Ensure feature consistency
    train_cols = set(X_train_processed.columns)
    test_cols = set(X_test_processed.columns)
    if train_cols != test_cols:
        common_cols = list(train_cols & test_cols)
        X_train_processed = X_train_processed[common_cols]
        X_test_processed = X_test_processed[common_cols]
        print(f"After correction - Common features count: {len(common_cols)}")

    # 返回原始特征（用于检查衍生特征的原始特征是否存在）
    return X_train_processed, X_test_processed, y_train, y_test

# --------------------------
# 3. 多特征集评估函数（修复valid_features逻辑）
# --------------------------
def evaluate_multiple_feature_sets(X_train, X_test, y_train, y_test, feature_sets, model_name="DecisionTree"):
    results = []

    # Define model
    if model_name == "DecisionTree":
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', DecisionTreeClassifier(class_weight='balanced'))
        ])
        params = {'model__max_depth': [5, 7, 10], 'model__criterion': ['gini', 'entropy']}
    elif model_name == "RandomForest":
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(class_weight='balanced', n_jobs=-1))
        ])
        params = {'model__n_estimators': [50, 100], 'model__max_depth': [5, 10]}
    elif model_name == "LogisticRegression":
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(class_weight='balanced', max_iter=1000))
        ])
        params = {'model__C': [0.1, 1, 10]}

    # 新增：XGBoost 模型配置
    elif model_name == "XGBoost":
        pos_weight = sum(y_train == 0) / sum(y_train == 1)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                random_state=42,
                scale_pos_weight=pos_weight  # 平衡正负样本
            ))])
        # XGBoost 特有的网格搜索参数
        params = {
            'model__learning_rate': [0.01, 0.1],
            'model__n_estimators': [200, 300],
            'model__max_depth': [5, 7, 9],
            'model__subsample': [0.8, 1.0],
            'model__colsample_bytree': [0.8, 1.0]
        }
    
    else:
        raise ValueError("Model name does not exist. Choose from: 'DecisionTree', 'RandomForest', 'LogisticRegression', 'XGBoost'")


    # 遍历每个特征集
    for name, features in feature_sets.items():
        print(f"\n===== Evaluating feature set: {name} =====")
        
        # 简化：只保留在X_train中存在的特征，不考虑映射
        valid_features = [f for f in features if f in X_train.columns]
        
        # 输出有效特征数量
        print(f"Feature set {name} - Original count: {len(features)}, Valid count: {len(valid_features)}")
        if not valid_features:
            print(f"Feature set {name} has no valid features, skipping evaluation")
            continue


        # 提取有效特征子集
        X_train_sub = X_train[valid_features]
        X_test_sub = X_test[valid_features]

        # 网格搜索优化模型
        gs = GridSearchCV(
            pipeline, param_grid=params,
            scoring='roc_auc', cv=3, n_jobs=-1, verbose=1
        )
        gs.fit(X_train_sub, y_train)
        best_model = gs.best_estimator_

        # 预测与评估
        y_pred = best_model.predict(X_test_sub)
        y_prob = best_model.predict_proba(X_test_sub)[:, 1]

        # 计算评估指标
        metrics = {
            'Feature Set': name,
            'Valid Features': len(valid_features),
            'Best Params': gs.best_params_,
            'Train AUC': cross_val_score(best_model, X_train_sub, y_train, cv=3, scoring='roc_auc').mean(),
            'Test AUC': roc_auc_score(y_test, y_prob),
            'Test Accuracy': accuracy_score(y_test, y_pred),
            'Test F1 Score': f1_score(y_test, y_pred),
        }
        results.append(metrics)

        # 输出详细评估结果
        print(f"Best params: {gs.best_params_}")
        print(f"Test AUC: {metrics['Test AUC']:.4f}")
        print(f"Test Accuracy: {metrics['Test Accuracy']:.4f}")
        print(f"Test F1 Score: {metrics['Test F1 Score']:.4f}")
        print("Classification Report:\n", classification_report(y_test, y_pred))

    # 可视化结果
    visualize_results(results)
    return pd.DataFrame(results)

# --------------------------
# 4. 结果可视化函数（折线图）
# --------------------------
def visualize_results(results):
    df = pd.DataFrame(results)
    
    # 设置风格
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)
    
    # 1. Test AUC折线图
    plt.figure()
    sns.lineplot(
        data=df,
        x='Feature Set',
        y='Test AUC',
        marker='o',  # 数据点标记
        linewidth=2,
        markersize=8
    )
    plt.title('Test Set AUC Comparison Across Feature Sets', fontsize=14)
    plt.xlabel('Feature Set', fontsize=12)
    plt.ylabel('AUC Score', fontsize=12)
    plt.ylim(0.5, 1.0)  # AUC范围固定
    plt.xticks(rotation=45)  # 特征集名称旋转45度避免重叠
    plt.tight_layout()
    plt.savefig('test_auc_comparison.png', dpi=300)
    plt.close()

    # 2. 多指标折线图（Accuracy, F1, AUC）
    df_melt = df.melt(
        id_vars='Feature Set',
        value_vars=['Test Accuracy', 'Test F1 Score', 'Test AUC'],
        var_name='Metric',
        value_name='Score'
    )
    
    plt.figure()
    sns.lineplot(
        data=df_melt,
        x='Feature Set',
        y='Score',
        hue='Metric',  # 按指标分组
        marker='o',
        linewidth=2,
        markersize=8
    )
    plt.title('Multi-metric Comparison Across Feature Sets', fontsize=14)
    plt.xlabel('Feature Set', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.savefig('multi_metric_comparison.png', dpi=300)
    plt.close()

    print("Visualizations saved as 'test_auc_comparison.png' and 'multi_metric_comparison.png'")
# --------------------------
# 5. 主函数
# --------------------------
def main():
    file_path = '/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/dataset/cleaned_loan_data.csv'
    
    # Define feature sets
    feature_sets = {
        'FS1_k30': [
            'sub_grade', 'int_rate', 'fico_range_low', 'fico_range_high', 'term', 
            'acc_open_past_24mths', 'inq_last_6mths', 'bc_open_to_buy', 'max_bal_bc', 
            'num_tl_op_past_12m', 'total_bc_limit', 'bc_util', 'all_util', 'percent_bc_gt_75', 
            'mths_since_rcnt_il', 'verification_status', 'initial_list_status', 'tot_hi_cred_lim', 
            'revol_util', 'disbursement_method', 'il_util', 'avg_cur_bal', 'mo_sin_rcnt_tl', 
            'mo_sin_rcnt_rev_tl_op', 'tot_cur_bal', 'open_act_il', 'mths_since_recent_bc', 
            'total_bal_il', 'num_actv_rev_tl', 'num_rev_tl_bal_gt_0'
        ],
        'FS2_k30': ['initial_list_status', 'emp_title', 'pct_tl_nvr_dlq', 'verification_status', 
                   'emp_length', 'term', 'num_tl_op_past_12m', 'int_rate', 'home_ownership', 'sub_grade', 
                   'inq_last_6mths', 'num_actv_bc_tl', 'percent_bc_gt_75', 'installment', 'acc_open_past_24mths',
                     'num_actv_rev_tl', 'num_rev_tl_bal_gt_0', 'mort_acc', 'num_bc_sats', 'fico_range_high', 
                     'fico_range_low', 'open_il_24m', 'open_il_12m', 'num_op_rev_tl', 'earliest_cr_line', 
                     'open_act_il', 'num_bc_tl', 'open_rv_12m', 'open_rv_24m', 'num_sats'],
        'FS3_k30': ['sub_grade', 'int_rate', 'term', 'fico_range_high', 'mths_since_rcnt_il',
                    'max_bal_bc', 'fico_range_low', 'acc_open_past_24mths', 'all_util', 'dti',
                      'num_tl_op_past_12m', 'bc_open_to_buy', 'total_bal_il', 'il_util', 'open_rv_24m', 'open_act_il',
                        'tot_hi_cred_lim', 'avg_cur_bal', 'installment', 'inq_last_6mths', 'annual_inc', 'funded_amnt',
                          'tot_cur_bal', 'total_bc_limit', 'loan_amnt', 'disbursement_method', 'verification_status', 
                          'mo_sin_rcnt_tl', 'mths_since_recent_bc', 'mo_sin_old_rev_tl_op'],
        'FS4_k60': ['loan_amnt', 'funded_amnt', 'term', 'int_rate', 'installment', 'sub_grade', 'emp_length', 
                   'annual_inc', 'dti', 'delinq_2yrs', 'fico_range_low', 'fico_range_high', 'inq_last_6mths', 
                   'mths_since_last_delinq', 'open_acc', 'revol_bal', 'revol_util', 'total_acc', 'mths_since_last_major_derog', 
                   'tot_coll_amt', 'tot_cur_bal', 'open_act_il', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_24m',
                     'max_bal_bc', 'all_util', 'inq_fi', 'inq_last_12m', 'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy',
                       'bc_util', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 
                       'mort_acc', 'mths_since_recent_bc', 'mths_since_recent_bc_dlq', 'mths_since_recent_inq', 
                       'mths_since_recent_revol_delinq', 'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl', 
                       'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0',
                         'num_sats', 'num_tl_op_past_12m', 'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'tot_hi_cred_lim', 
                         'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit', 'earliest_cr_line_year',
                           'earliest_cr_line_month', 'emp_title_cluster', 'emp_title_cluster_freq']
    }

    # Generate processed data
    X_train, X_test, y_train, y_test = create_data_pipeline(file_path)

    # Evaluate all feature sets
    results_df = evaluate_multiple_feature_sets(
        X_train, X_test, y_train, y_test,
        feature_sets=feature_sets,
        model_name="XGBoost"  # 可以选择 'DecisionTree', 'RandomForest', 'LogisticRegression', 'XGBoost'
    )

    # Save results
    results_df.to_csv('feature_set_comparison_results.csv', index=False)
    print("\nAll evaluation results saved to 'feature_set_comparison_results.csv'")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

# 读取数据
data = pd.read_csv('/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/dataset/train_data_encoded.csv')

# 定义特征映射关系
def get_original_feature_mapping():
    return {
        'earliest_cr_line_year': 'earliest_cr_line', 'earliest_cr_line_month': 'earliest_cr_line',
        'last_credit_pull_d_year': 'last_credit_pull_d', 'last_credit_pull_d_month': 'last_credit_pull_d',
        'home_ownership_': 'home_ownership',
        'verification_status_': 'verification_status',
        'emp_title_cluster': 'emp_title', 'emp_title_cluster_freq': 'emp_title',
    }

# 特征分组函数
def group_features_by_original(X, mapping):
    groups = {}
    for col in X.columns:
        original = mapping.get(col) or next(
            (orig for prefix, orig in mapping.items() if prefix.endswith('_') and col.startswith(prefix)),
            col
        )
        groups.setdefault(original, []).append(col)
    return groups

# 移除常量特征
def remove_constant_features(X):
    constant_features = X.columns[X.nunique() == 1].tolist()
    if constant_features:
        X = X.drop(columns=constant_features)
        print(f"移除 {len(constant_features)} 个常量特征")
    return X

# 3.1 过滤法（ANOVA和互信息）
def filter_based_selection(X, y, feature_groups, k):
    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    anova_scores, mi_scores = {}, {}
    
    for orig, derivs in feature_groups.items():
        valid_features = [f for f in derivs if f in X.columns]
        if not valid_features:
            anova_scores[orig] = mi_scores[orig] = 0
            continue
        try:
            anova_scores[orig] = np.mean([f_classif(X_scaled[[f]], y)[0][0] for f in valid_features])
        except:
            anova_scores[orig] = 0
        try:
            mi_scores[orig] = np.mean([mutual_info_classif(X[[f]], y, random_state=42)[0] for f in valid_features])
        except:
            mi_scores[orig] = 0
    
    sorted_anova = sorted(anova_scores.items(), key=lambda x: x[1], reverse=True)
    sorted_mi = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 可视化Top-k（仅当k为最终选择值时生成）
    if k == best_k:
        plt.figure(figsize=(20, 8))
        sns.barplot(x=[s for _, s in sorted_anova[:k]], y=[f for f, _ in sorted_anova[:k]])
        plt.title(f'Top {k} Feature Groups by ANOVA')
        plt.tight_layout()
        plt.savefig('anova_topk.png', dpi=300)
        plt.close()
        
        plt.figure(figsize=(20, 8))
        sns.barplot(x=[s for _, s in sorted_mi[:k]], y=[f for f, _ in sorted_mi[:k]])
        plt.title(f'Top {k} Feature Groups by MI')
        plt.tight_layout()
        plt.savefig('mi_topk.png', dpi=300)
        plt.close()
    
    return {
        'anova_features': [f for g, _ in sorted_anova[:k] for f in feature_groups[g] if f in X.columns],
        'mi_features': [f for g, _ in sorted_mi[:k] for f in feature_groups[g] if f in X.columns]
    }

# 3.2 嵌入法（随机森林）
def embedding_based_selection(X, y, feature_groups, k):
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    group_importance = {}
    for orig, derivs in feature_groups.items():
        valid_features = [f for f in derivs if f in X.columns]
        if valid_features:
            group_importance[orig] = np.mean([rf.feature_importances_[X.columns.get_loc(f)] for f in valid_features])
        else:
            group_importance[orig] = 0
    
    sorted_groups = sorted(group_importance.items(), key=lambda x: x[1], reverse=True)
    
    if k == best_k:
        plt.figure(figsize=(12, 8))
        sns.barplot(x=[s for _, s in sorted_groups[:k]], y=[f for f, _ in sorted_groups[:k]])
        plt.title(f'Top {k} Feature Groups by RF Importance')
        plt.tight_layout()
        plt.savefig('rf_topk.png', dpi=300)
        plt.close()
    
    return [f for g, _ in sorted_groups[:k] for f in feature_groups[g] if f in X.columns]

# 3.3 包裹法（RFECV自动确定最佳特征数）
def wrapper_based_selection(X, y, feature_groups):
    estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rfecv = RFECV(
        estimator, step=1, cv=StratifiedKFold(3),
        scoring='roc_auc', min_features_to_select=5, n_jobs=-1
    )
    
    # 采样加速
    X_sample, _, y_sample, _ = train_test_split(X, y, test_size=0.7, random_state=42)
    rfecv.fit(X_sample, y_sample)
    
    # 可视化特征数与AUC关系
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.xlabel('Number of Features')
    plt.ylabel('AUC')
    plt.title('RFECV Feature Selection')
    plt.savefig('rfecv_curve.png', dpi=300)
    plt.close()
    
    selected_features = X.columns[rfecv.support_].tolist()
    print(f"包裹法最优特征数: {len(selected_features)}")
    return selected_features


# 4. 数据驱动确定最佳k值（核心）

# 定义评估函数：用交叉验证测试不同k值的性能
def evaluate_k_performance(X, y, feature_groups, k_range, method='filter_anova'):
    results = []
    for k in k_range:
        # 根据方法获取特征
        if method == 'filter_anova':
            features = filter_based_selection(X, y, feature_groups, k)['anova_features']
        elif method == 'filter_mi':
            features = filter_based_selection(X, y, feature_groups, k)['mi_features']
        elif method == 'embedding':
            features = embedding_based_selection(X, y, feature_groups, k)
        else:
            raise ValueError("方法不存在")
        
        # 避免特征为空
        if not features:
            continue
        
        # 用逻辑回归评估（作为基准模型）
        X_sub = X[features]
        model = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(class_weight='balanced', max_iter=1000))])
        cv_scores = cross_val_score(model, X_sub, y, cv=5, scoring='roc_auc')
        
        results.append({
            'k': k,
            'mean_auc': cv_scores.mean(),
            'std_auc': cv_scores.std(),
            'feature_count': len(features)
        })
    
    return pd.DataFrame(results)

# 定义k值范围（根据特征分组数调整）
k_range = range(10, min(100, len(feature_groups)), 5)  # 从10到100，步长5
print(f"测试k值范围: {list(k_range)}")

# 分别评估三种方法的不同k值
print("\n===== 评估过滤法(ANOVA) =====")
filter_anova_results = evaluate_k_performance(X, y, feature_groups, k_range, method='filter_anova')

print("\n===== 评估过滤法(互信息) =====")
filter_mi_results = evaluate_k_performance(X, y, feature_groups, k_range, method='filter_mi')

print("\n===== 评估嵌入法 =====")
embedding_results = evaluate_k_performance(X, y, feature_groups, k_range, method='embedding')

# 确定每种方法的最佳k值（AUC最高）
best_k_anova = filter_anova_results.loc[filter_anova_results['mean_auc'].idxmax()]['k']
best_k_mi = filter_mi_results.loc[filter_mi_results['mean_auc'].idxmax()]['k']
best_k_embedding = embedding_results.loc[embedding_results['mean_auc'].idxmax()]['k']

best_k = best_k_anova  # 可根据实际结果选择最优方法的k值
print(f"\n过滤法(ANOVA)最佳k值: {best_k_anova}, 对应AUC: {filter_anova_results['mean_auc'].max():.4f}")
print(f"过滤法(互信息)最佳k值: {best_k_mi}, 对应AUC: {filter_mi_results['mean_auc'].max():.4f}")
print(f"嵌入法最佳k值: {best_k_embedding}, 对应AUC: {embedding_results['mean_auc'].max():.4f}")

# 可视化不同k值的性能曲线
plt.figure(figsize=(12, 6))
plt.errorbar(filter_anova_results['k'], filter_anova_results['mean_auc'], yerr=filter_anova_results['std_auc'], label='ANOVA')
plt.errorbar(filter_mi_results['k'], filter_mi_results['mean_auc'], yerr=filter_mi_results['std_auc'], label='Mutual Info')
plt.errorbar(embedding_results['k'], embedding_results['mean_auc'], yerr=embedding_results['std_auc'], label='Random Forest')
plt.xlabel('k (Number of Feature Groups)')
plt.ylabel('Mean AUC (5-fold CV)')
plt.title('Feature Group Count vs. Model Performance')
plt.legend()
plt.grid(True)
plt.savefig('k_performance_curve.png', dpi=300)
plt.close()

# --------------------------
# 5. 获取最终特征集并保存
# --------------------------
# 过滤法(ANOVA)最终特征
final_filter_anova = filter_based_selection(X, y, feature_groups, best_k_anova)['anova_features']
pd.DataFrame(final_filter_anova, columns=['Feature']).to_csv('final_anova_features.csv', index=False)

# 过滤法(互信息)最终特征
final_filter_mi = filter_based_selection(X, y, feature_groups, best_k_mi)['mi_features']
pd.DataFrame(final_filter_mi, columns=['Feature']).to_csv('final_mi_features.csv', index=False)

# 嵌入法最终特征
final_embedding = embedding_based_selection(X, y, feature_groups, best_k_embedding)
pd.DataFrame(final_embedding, columns=['Feature']).to_csv('final_embedding_features.csv', index=False)

# 包裹法最终特征（单独处理，自动确定特征数）
final_wrapper = wrapper_based_selection(X, y, feature_groups)
pd.DataFrame(final_wrapper, columns=['Feature']).to_csv('final_wrapper_features.csv', index=False)

print("\n所有方法的最佳特征集已保存至CSV文件")
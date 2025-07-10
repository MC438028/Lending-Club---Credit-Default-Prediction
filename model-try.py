import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# # 读取数据文件
# data = pd.read_csv('/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/train_data_encoded.csv')

# no_meaning_features = ['id']
# data = data.drop(columns=no_meaning_features, errors='ignore')

# # 定义原始特征与衍生特征的映射关系（根据特征工程逻辑）
# def get_original_feature_mapping():
#     """建立衍生特征到原始特征的映射关系"""
#     mapping = {
#         # 日期衍生特征
#         'issue_d_year': 'issue_d',
#         'issue_d_month': 'issue_d',
#         'last_pymnt_d_year': 'last_pymnt_d',
#         'last_pymnt_d_month': 'last_pymnt_d',
#         'earliest_cr_line_year': 'earliest_cr_line',
#         'earliest_cr_line_month': 'earliest_cr_line',
#         'last_credit_pull_d_year': 'last_credit_pull_d',
#         'last_credit_pull_d_month': 'last_credit_pull_d',
#         'next_pymnt_d_has_date': 'next_pymnt_d',
        
#         # One-Hot编码特征（前缀匹配）
#         'home_ownership_': 'home_ownership',
#         'verification_status_': 'verification_status',
#         'purpose_': 'purpose',
        
#         # 聚类衍生特征
#         'emp_title_cluster': 'emp_title',
#         'emp_title_cluster_freq': 'emp_title',
#         'title_cluster': 'title',
#         'title_cluster_freq': 'title'
#     }
#     return mapping

# # 按原始特征分组
# def group_features_by_original(X, mapping):
#     """将衍生特征按原始特征分组"""
#     groups = {}
#     # 先添加所有未映射的特征（原始特征）
#     for col in X.columns:
#         original = None
#         # 精确匹配
#         if col in mapping:
#             original = mapping[col]
#         # 前缀匹配（处理One-Hot编码特征）
#         else:
#             for prefix, orig in mapping.items():
#                 if prefix.endswith('_') and col.startswith(prefix):
#                     original = orig
#                     break
#         # 未匹配的视为原始特征
#         if original is None:
#             original = col
            
#         if original not in groups:
#             groups[original] = []
#         groups[original].append(col)
#     return groups

# # 添加常量特征过滤函数
# def remove_constant_features(X):
#     unique_counts = X.nunique()
#     constant_features = unique_counts[unique_counts == 1].index.tolist()
#     if constant_features:
#         print(f"发现{len(constant_features)}个常量特征，将被移除: {constant_features}")
#         X = X.drop(columns=constant_features)
#     else:
#         print("未发现常量特征")
#     return X

# # 处理特征和标签
# if 'loan_status_flag' in data.columns:
#     X = data.drop(['loan_status', 'loan_status_flag'], axis=1, errors='ignore')
#     y = data['loan_status_flag']
# else:
#     X = data.drop('loan_status', axis=1, errors='ignore')
#     loan_status_flag = [
#         'Charged Off', 'Default', 
#         'Does not meet the credit policy. Status:Charged Off',
#         'Late (31-120 days)'
#     ]
#     y = data['loan_status'].apply(lambda x: 1 if x in loan_status_flag else 0)

# # 只保留数值特征
# X = X.select_dtypes(include=[np.number])
# print(f"处理后特征数量: {X.shape[1]}")

# X = remove_constant_features(X)
# print(f"移除常量特征后特征数量: {X.shape[1]}")

# # 获取特征分组
# feature_mapping = get_original_feature_mapping()
# feature_groups = group_features_by_original(X, feature_mapping)
# print(f"\n特征分组数量: {len(feature_groups)}")
# print("特征分组示例:")
# for i, (orig, derivs) in enumerate(list(feature_groups.items())[:5]):
#     print(f"  原始特征 {orig}: {len(derivs)} 个衍生特征")

# # Method1: 过滤法特征选择（按组评估，保留ANOVA和互信息）
# def filter_based_selection(X, y, feature_groups, k=30):
#     """按原始特征组进行过滤法选择，同时计算ANOVA和互信息两种评分"""
#     # 标准化特征（用于ANOVA）
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
#     # 初始化两组评分字典
#     anova_group_scores = {}
#     mi_group_scores = {}
    
#     # 计算每个特征组的平均分数（ANOVA和互信息）
#     for orig_feature, deriv_features in feature_groups.items():
#         if not deriv_features:
#             continue
        
#         # 提取组内有效特征（确保特征存在于X中）
#         valid_features = [f for f in deriv_features if f in X.columns]
#         if not valid_features:
#             anova_group_scores[orig_feature] = 0
#             mi_group_scores[orig_feature] = 0
#             continue
        
#         # 计算ANOVA分数（基于标准化数据）
#         try:
#             f_scores = f_classif(X_scaled_df[valid_features], y)[0]
#             anova_group_scores[orig_feature] = np.mean(f_scores)
#         except:
#             anova_group_scores[orig_feature] = 0
        
#         # 计算互信息分数（基于原始数据，无需标准化）
#         try:
#             mi_scores = mutual_info_classif(X[valid_features], y, random_state=42)
#             mi_group_scores[orig_feature] = np.mean(mi_scores)
#         except:
#             mi_group_scores[orig_feature] = 0
    
#     # 按两种评分分别排序并选择top组
#     sorted_anova_groups = sorted(anova_group_scores.items(), key=lambda x: x[1], reverse=True)
#     sorted_mi_groups = sorted(mi_group_scores.items(), key=lambda x: x[1], reverse=True)
    
#     top_anova_groups = [g[0] for g in sorted_anova_groups[:k]]
#     top_mi_groups = [g[0] for g in sorted_mi_groups[:k]]
    
#     # 提取对应的特征列表
#     top_anova_features = [f for g in top_anova_groups for f in feature_groups[g] if f in X.columns]
#     top_mi_features = [f for g in top_mi_groups for f in feature_groups[g] if f in X.columns]
    
#     # 可视化两种评分的top特征组
#     plt.figure(figsize=(20, 8))
    
#     # ANOVA结果可视化
#     plt.subplot(1, 2, 1)
#     anova_df = pd.DataFrame(sorted_anova_groups[:30], columns=['Original Feature', 'Avg ANOVA Score'])
#     sns.barplot(x='Avg ANOVA Score', y='Original Feature', data=anova_df)
#     plt.title('Top 30 Feature Groups by Average ANOVA Score')
    
#     # 互信息结果可视化
#     plt.subplot(1, 2, 2)
#     mi_df = pd.DataFrame(sorted_mi_groups[:30], columns=['Original Feature', 'Avg MI Score'])
#     sns.barplot(x='Avg MI Score', y='Original Feature', data=mi_df)
#     plt.title('Top 30 Feature Groups by Average Mutual Information Score')
    
#     plt.tight_layout()
#     plt.savefig('feature_group_selection_filter.png', dpi=300)
#     plt.close()
    
#     print(f"ANOVA选择了 {len(top_anova_groups)} 个特征组，共 {len(top_anova_features)} 个特征")
#     print(f"互信息选择了 {len(top_mi_groups)} 个特征组，共 {len(top_mi_features)} 个特征")
#     return {
#         # ANOVA相关结果
#         'anova_top_groups': top_anova_groups,
#         'anova_top_features': top_anova_features,
#         'anova_group_scores': anova_group_scores,
#         # 互信息相关结果
#         'mi_top_groups': top_mi_groups,
#         'mi_top_features': top_mi_features,
#         'mi_group_scores': mi_group_scores,
#         # 原始排序结果
#         'sorted_anova_groups': sorted_anova_groups,
#         'sorted_mi_groups': sorted_mi_groups
#     }

# # 执行过滤法特征选择（使用特征组）
# filter_results = filter_based_selection(X, y, feature_groups, k=30)

# # 保存结果（分别保存ANOVA和互信息的特征组）
# anova_group_df = pd.DataFrame(
#     filter_results['sorted_anova_groups'],
#     columns=['Original Feature', 'Average ANOVA Score']
# )
# anova_group_df.to_csv('anova_based_feature_groups.csv', index=False)

# mi_group_df = pd.DataFrame(
#     filter_results['sorted_mi_groups'],
#     columns=['Original Feature', 'Average MI Score']
# )
# mi_group_df.to_csv('mi_based_feature_groups.csv', index=False)

# print("ANOVA特征组结果已保存至 anova_based_feature_groups.csv")
# print("互信息特征组结果已保存至 mi_based_feature_groups.csv")
# # 执行过滤法特征选择
# filter_results = filter_based_selection(X, y, feature_groups, k=30)


# # Method 2：嵌入法特征选择（基于树模型，按组评估）
# # 在model.py中，调用embedding_based_selection前添加
# def embedding_based_selection(X, y, feature_groups):
#     # 训练随机森林
#     rf = RandomForestClassifier(
#         n_estimators=100,
#         max_depth=10,
#         random_state=42,
#         n_jobs=-1
#     )
#     rf.fit(X, y)
    
#     # 计算每个特征组的平均重要性
#     group_importance = {}
#     for orig_feature, deriv_features in feature_groups.items():
#         if not deriv_features or orig_feature not in [f for f, _ in feature_groups.items()]:
#             continue
#         # 提取组内特征的重要性
#         importances = [rf.feature_importances_[X.columns.get_loc(f)] for f in deriv_features]
#         group_importance[orig_feature] = np.mean(importances)
    
#     # 排序特征组
#     sorted_groups = sorted(group_importance.items(), key=lambda x: x[1], reverse=True)
    
#     # 可视化
#     plt.figure(figsize=(12, 8))
#     importance_df = pd.DataFrame(sorted_groups[:30], columns=['Original Feature', 'Avg Importance'])
#     sns.barplot(x='Avg Importance', y='Original Feature', data=importance_df)
#     plt.title('Top 30 Feature Groups by Random Forest Importance')
#     plt.tight_layout()
#     plt.savefig('feature_group_selection_embedding.png', dpi=300)
#     plt.close()
    
#     return {
#         'sorted_groups': sorted_groups,
#         'group_importance': group_importance
#     }

# # 执行嵌入法特征选择
# embedding_results = embedding_based_selection(X, y, feature_groups)
# embedding_df = pd.DataFrame(
#     embedding_results['sorted_groups'], 
#     columns=['Original Feature', 'Average Importance']
# )
# embedding_df.to_csv('embedding_based_feature_groups.csv', index=False)
# print("特征组重要性结果已保存至 embedding_based_feature_groups.csv")

# # 3. 包裹法特征选择（递归特征消除，按组进行）
# def wrapper_based_selection(X, y, feature_groups):
#     # 初始化模型
#     estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    
#     # 准备特征组映射（用于RFE的特征分组）
#     group_list = [feature_groups[g] for g in feature_groups if feature_groups[g]]
#     group_indices = []
#     for group in group_list:
#         group_indices.extend([X.columns.get_loc(f) for f in group])
    
#     # 递归特征消除（按组进行）
#     rfecv = RFECV(
#         estimator=estimator,
#         step=1,  # 每次移除一个最差的组
#         cv=StratifiedKFold(5),
#         scoring='roc_auc',
#         min_features_to_select=5,
#         n_jobs=-1
#     )
    
#     # 只使用部分数据加速计算
#     sample_size = min(10000, len(X))
#     X_sample = X.sample(sample_size, random_state=42)
#     y_sample = y.loc[X_sample.index]
    
#     rfecv.fit(X_sample, y_sample)
    
#     # 确定选中的特征组
#     selected_indices = rfecv.support_
#     selected_features = X_sample.columns[selected_indices].tolist()
    
#     # 反向映射到原始特征组
#     selected_groups = set()
#     for f in selected_features:
#         for orig, derivs in feature_groups.items():
#             if f in derivs:
#                 selected_groups.add(orig)
#                 break
    
#     print(f"最优特征组数量: {len(selected_groups)}")
#     print(f"选中的特征组: {selected_groups}")
    
#     # 可视化交叉验证分数
#     plt.figure(figsize=(12, 8))
#     scores = rfecv.cv_results_['mean_test_score']
#     plt.plot(range(1, len(scores) + 1), scores)
#     plt.xlabel('Number of feature groups selected')
#     plt.ylabel('Cross validation score (ROC AUC)')
#     plt.title('Feature Group Selection with RFECV')
#     plt.tight_layout()
#     plt.savefig('feature_group_selection_rfec.png')
#     plt.close()
    
#     return {
#         'selected_groups': selected_groups,
#         'selected_features': selected_features,
#         'cv_scores': scores
#     }

# wrapper_results = wrapper_based_selection(X, y, feature_groups)
# pd.DataFrame(wrapper_results['selected_features'], columns=['Feature']).to_csv('wrapper_selected_features.csv', index=False)
# print("包裹法选择的特征已保存至 wrapper_selected_features.csv")

# 定义分类函数（从已有代码中提取）
def categorize_columns(df):
    columns_dict = {
        '贷款核心属性类(贷款基本条款)': ['id', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate',
                            'installment', 'grade', 'sub_grade', 'initial_list_status', 'issue_d',
                            'disbursement_method'],
        '借款人信息类(个人背景特征)': ['emp_title', 'emp_length', 'home_ownership', 'annual_inc', 'addr_state',
                           'application_type', 'verification_status'],
        '信用评估类(核心风险指标)': ['fico_range_low', 'fico_range_high', 'last_fico_range_high', 'last_fico_range_low',
                           'pub_rec', 'pub_rec_bankruptcies', 'tax_liens', 'earliest_cr_line', 'last_credit_pull_d',
                           'inq_last_6mths', 'inq_fi', 'inq_last_12m', 'pub_rec', 'total_cu_tl', 'all_util',
                           'il_util', 'bc_util', 'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'tot_hi_cred_lim',
                           'total_bc_limit', 'total_il_high_credit_limit'],
        '还款能力类': ['annual_inc', 'dti', 'tot_coll_amt', 'tot_cur_bal', 'total_bal_ex_mort', 'total_rev_hi_lim',
                 'payment_plan', 'revol_bal', 'revol_util', 'avg_cur_bal', 'bc_open_to_buy', 'total_bal_il', 'tot_cur_bal'],
        '违约强相关类(贷款违约信息)': ['loan_status', 'delinq_2yrs', 'mths_since_last_delinq', 'acc_now_delinq',
                           'delinq_amnt', 'num_accts_ever_120_pd', 'num_tl_120dpd_2m', 'num_tl_30dpd',
                           'num_tl_90g_dpd_24m', 'debt_settlement_flag', 'hardship_flag', 'recoveries',
                           'collection_recovery_fee', 'collections_12_mths_ex_med', 'mths_since_last_major_derog',
                           'chargeoff_within_12_mths', 'mths_since_recent_bc_dlq', 'mths_since_recent_revol_delinq'],
        '还款活动类': ['last_pymnt_d', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
                 'last_pymnt_amnt', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee'],
        '账户活动类': ['open_acc', 'total_acc', 'open_acc_6m', 'open_act_il', 'open_il_12m', 'open_il_24m', 'open_rv_12m',
                  'open_rv_24m', 'mths_since_rcnt_il', 'total_bal_il', 'max_bal_bc', 'open_rv_12m', 'open_rv_24m',
                  'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl',
                  'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_op_past_12m', 'acc_open_past_24mths',
                  'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc',
                  'mths_since_recent_bc', 'mths_since_recent_inq', 'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m'],
        '其他相关类': ['title', 'zip_code']
    }
    return columns_dict

def analyze_selected_features(csv_file_path, k=30):
    base_path = '/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/'
    full_path = base_path + csv_file_path
    selected_features_df = pd.read_csv(full_path)
    
    if 'Feature' in selected_features_df.columns:
        selected_features = selected_features_df['Feature'].tolist()[:k]
    elif 'Original Feature' in selected_features_df.columns:
        selected_features = selected_features_df['Original Feature'].tolist()[:k]
    else:
        print("未找到合适的特征列。")
        return
    
    category_dict = categorize_columns(None)
    category_selected_features = {}
    for category, features in category_dict.items():
        category_selected_features[category] = [f for f in features if f in selected_features]
    
    for category, selected in category_selected_features.items():
        print(f"{category}: {selected}")
    
    return category_selected_features

# 示例调用
anova_csv = 'anova_based_feature_groups.csv'
mi_csv = 'mi_based_feature_groups.csv'
embedding_csv = 'embedding_based_feature_groups.csv'
wrapper_csv = 'wrapper_selected_features.csv'

print("ANOVA 特征筛选结果统计：")
analyze_selected_features(anova_csv)
print("\n互信息特征筛选结果统计：")
analyze_selected_features(mi_csv)
print("\n嵌入法特征筛选结果统计：")
analyze_selected_features(embedding_csv)
print("\n包裹法特征筛选结果统计：")
analyze_selected_features(wrapper_csv)
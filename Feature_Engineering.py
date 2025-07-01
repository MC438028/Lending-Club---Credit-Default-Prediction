import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# 假设数据已经经过清洗和预处理，读取处理后的数据
cleaned_data = pd.read_csv('/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/processed_loan_data.csv', low_memory=False)

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
        '其他相关类': ['title', 'zip_code', 'policy_code']
    }
    return columns_dict

columns_dict = categorize_columns(cleaned_data)

def check_columns_exist(cleaned_data, columns):
    """检查列是否存在并返回存在的列列表"""
    return [col for col in columns if col in cleaned_data.columns]

# Binary encoding 处理二分类变量
def process_binary_columns(cleaned_data):
    # 手动映射明确的列
    manual_mappings = {
        'term': {' 36 months': 0, ' 60 months': 1},
        'pymnt_plan': {'n': 0, 'y': 1},
        'application_type': {'Individual': 0, 'Joint App': 1}
    }
    
    for col, mapping in manual_mappings.items():
        if col in cleaned_data.columns:
            cleaned_data[col] = cleaned_data[col].map(mapping)
    
    # 自动映射剩余列
    remaining_cols = ['initial_list_status', 'disbursement_method']
    existing_remaining = [col for col in remaining_cols if col in cleaned_data.columns]
    
    le = LabelEncoder()
    for col in existing_remaining:
        cleaned_data[col] = le.fit_transform(cleaned_data[col].astype(str))
    
    print("\nBinary Encoding 处理结果:")
    print(cleaned_data[list(manual_mappings.keys()) + existing_remaining].head())
    return cleaned_data 

# One Hot encoding 处理分类变量，修改为替换原列而非新增
def process_categorical_columns(cleaned_data):
    categorical_cols = ['home_ownership', 'verification_status', 'purpose']
    existing_cols = check_columns_exist(cleaned_data, categorical_cols)
    
    # 执行One-Hot编码，直接替换原列
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cols = ohe.fit_transform(cleaned_data[existing_cols])
    encoded_df = pd.DataFrame(encoded_cols, columns=ohe.get_feature_names_out(existing_cols))
    
    # 特征映射关系
    print("\n特征映射关系:")
    all_feature_names = ohe.get_feature_names_out(existing_cols)
    start_idx = 0
    for i, col in enumerate(existing_cols):
        num_categories = len(ohe.categories_[i])
        end_idx = start_idx + num_categories
        feature_names = all_feature_names[start_idx:end_idx]
        print(f"{col} 映射为: {feature_names.tolist()}")
        start_idx = end_idx
    
    # 替换原列并合并数据
    cleaned_data = cleaned_data.drop(existing_cols, axis=1)
    cleaned_data = pd.concat([cleaned_data, encoded_df], axis=1)
    
    print("\nOne-Hot Encoding 处理后数据示例:")
    print(cleaned_data[encoded_df.columns].head())
    return cleaned_data
    

# Label encoding 处理分类变量
def process_grade(cleaned_data):
    if 'grade' in cleaned_data.columns:
        grade_values = sorted(cleaned_data['grade'].dropna().unique())
        grade_map = {grade: i for i, grade in enumerate(grade_values)}
        cleaned_data['grade'] = cleaned_data['grade'].map(grade_map).fillna(-1).astype(int)
        print(f"\ngrade 标签编码映射: {grade_map}")
        print("grade 编码后示例:", cleaned_data['grade'].head())
    
    if 'sub_grade' in cleaned_data.columns:
        sub_grade_values = sorted(cleaned_data['sub_grade'].dropna().unique())
        sub_grade_map = {sub_grade: i for i, sub_grade in enumerate(sub_grade_values)}
        cleaned_data['sub_grade'] = cleaned_data['sub_grade'].map(sub_grade_map).fillna(-1).astype(int)
        print(f"\nsub_grade 标签编码映射: {sub_grade_map}")
        print("sub_grade 编码后示例:", cleaned_data['sub_grade'].head())
    return cleaned_data

def process_clustering_encoding(cleaned_data):
    # 处理 emp_title 列
    if 'emp_title' in cleaned_data.columns:
        vectorizer_emp = TfidfVectorizer(max_features=1000, stop_words='english')
        emp_title_tfidf = vectorizer_emp.fit_transform(cleaned_data['emp_title'])
        
        kmeans_emp = KMeans(n_clusters=30, random_state=42)
        cleaned_data['emp_title_cluster'] = kmeans_emp.fit_predict(emp_title_tfidf)
        
        print("\nemp_title 聚类编码结果:")
        for cluster in range(min(5, kmeans_emp.n_clusters)):
            samples = cleaned_data[cleaned_data['emp_title_cluster'] == cluster]['emp_title'].sample(
                min(3, len(cleaned_data[cleaned_data['emp_title_cluster'] == cluster])), random_state=42)
            print(f"聚类 {cluster} 示例: {list(samples)}")
        
        emp_cluster_freq = cleaned_data['emp_title_cluster'].value_counts(normalize=True).to_dict()
        cleaned_data['emp_title_cluster_freq'] = cleaned_data['emp_title_cluster'].map(emp_cluster_freq)
        cleaned_data = cleaned_data.drop('emp_title', axis=1)
    
    # 处理 title 列
    if 'title' in cleaned_data.columns:
        cleaned_data['title'] = cleaned_data['title'].fillna('unknown')
        vectorizer_title = TfidfVectorizer(max_features=500, stop_words='english')
        title_tfidf = vectorizer_title.fit_transform(cleaned_data['title'])
        
        kmeans_title = KMeans(n_clusters=20, random_state=42)
        cleaned_data['title_cluster'] = kmeans_title.fit_predict(title_tfidf)
        
        print("\ntitle 聚类编码结果:")
        for cluster in range(min(5, kmeans_title.n_clusters)):
            samples = cleaned_data[cleaned_data['title_cluster'] == cluster]['title'].sample(
                min(3, len(cleaned_data[cleaned_data['title_cluster'] == cluster])), random_state=42)
            print(f"聚类 {cluster} 示例: {list(samples)}")
        
        title_cluster_freq = cleaned_data['title_cluster'].value_counts(normalize=True).to_dict()
        cleaned_data['title_cluster_freq'] = cleaned_data['title_cluster'].map(title_cluster_freq)
        cleaned_data = cleaned_data.drop('title', axis=1)
    return cleaned_data


# 执行特征处理并打印结果
print("=== 开始特征工程处理 ===")

cleaned_data = process_binary_columns(cleaned_data)
print("\n=== Binary Encoding 处理完成 ===")

cleaned_data = process_categorical_columns(cleaned_data)
print("\n=== One-Hot Encoding 处理完成 ===")

cleaned_data = process_grade(cleaned_data)
print("\n=== Grade 标签编码处理完成 ===")

cleaned_data = process_clustering_encoding(cleaned_data)
print("\n=== 聚类编码处理完成 ===")

print("\n=== 特征工程全部完成 ===")
print("最终数据维度:", cleaned_data.shape)

# 保存处理后数据的前200行
if len(cleaned_data) > 200:
    # 若数据行数超过200，只保存前200行
    sampled_data = cleaned_data.head(200)
else:
    # 若数据行数不足200，保存全部数据
    sampled_data = cleaned_data

new_file_path = '/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/processed_loan_data_encoded_sample.csv'
sampled_data.to_csv(new_file_path, index=False)
print(f"\n特征工程后的数据前200行已保存至: {new_file_path}")

# # 筛选特征
# all_features = []
# for category, features in columns_dict.items():
#     existing_features = [f for f in features if f in cleaned_data.columns]
#     all_features.extend(existing_features)

# # 移除目标变量（假设为 'loan_status'）
# target_variable = 'loan_status'
# if target_variable in all_features:
#     all_features.remove(target_variable)

# X = cleaned_data[all_features]
# y = cleaned_data[target_variable]

# # 特征重要性分析与相关性计算
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split

# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 创建随机森林模型
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X_train, y_train)

# # 计算特征重要性
# feature_importances = pd.DataFrame({
#     'Feature': X.columns,
#     'Importance': rf.feature_importances_
# })
# feature_importances = feature_importances.sort_values('Importance', ascending=False)

# # 选择重要性最高的20个特征
# top_20_features = feature_importances.head(20)['Feature'].tolist()

# # 提取这些特征的数据
# top_data = cleaned_data[top_20_features]

# # 计算相关系数矩阵
# correlation_matrix = top_data.corr()

# # 绘制热力图
# plt.figure(figsize=(12, 10))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Top 20 Feature Correlation Heatmap')
# plt.tight_layout()
# plt.show()
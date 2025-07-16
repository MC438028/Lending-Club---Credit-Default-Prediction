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
from data_preprocessing import check_columns_exist

# 假设数据已经经过清洗和预处理，读取处理后的数据
train_data = pd.read_csv('/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML//dataset/train_data_preprocessed.csv', low_memory=False)

# Binary encoding 处理二分类变量
def process_binary_columns(train_data):
    # 手动映射明确的列
    manual_mappings = {
        'term': {' 36 months': 0, ' 60 months': 1},
    }
    
    for col, mapping in manual_mappings.items():
        if col in train_data.columns:
            train_data[col] = train_data[col].map(mapping)
    
    # 自动映射剩余列
    remaining_cols = ['initial_list_status', 'disbursement_method']
    existing_remaining = [col for col in remaining_cols if col in train_data.columns]
    
    le = LabelEncoder()
    for col in existing_remaining:
        train_data[col] = le.fit_transform(train_data[col].astype(str))
    
    # 新增loan_status_flag列：标记是否违约
    loan_status_flag = [
        'Charged Off',
        'Default',
        'Does not meet the credit policy. Status:Charged Off',
        'Late (31-120 days)'
    ]
    
    # 确保loan_status列存在再处理
    if 'loan_status' in train_data.columns:
        print("  创建loan_status_flag列...")
        train_data['loan_status_flag'] = train_data['loan_status'].apply(lambda x: 1 if x in loan_status_flag else 0)
        
        # 打印违约样本分布
        default_dist = train_data['loan_status_flag'].value_counts()
        print(f"  违约样本分布：\n{default_dist}")
        if len(default_dist) > 1:
            print(f"  违约率：{default_dist[1] / default_dist.sum():.2%}")
        else:
            print(f"  违约率：0.00%（所有样本均为非违约）")
    else:
        print("  警告: 未找到loan_status列，创建全0的loan_status_flag列")
        train_data['loan_status_flag'] = 0
    
    # 打印处理结果
    print("\nBinary Encoding 处理结果:")
    display_cols = list(manual_mappings.keys()) + existing_remaining + ['loan_status_flag']
    print(train_data[display_cols].head())
    
    print("=== 二分类列处理完成 ===\n")
    return train_data

#  # One Hot encoding 处理分类变量，修改为替换原列而非新增
# def process_categorical_columns(train_data):
#     """保持One-Hot Encoding逻辑，但以向量形式展示特征映射"""
#     categorical_cols = ['home_ownership', 'verification_status', 'purpose']
#     existing_cols = check_columns_exist(train_data, categorical_cols)
    
#     # 执行One-Hot编码
#     ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
#     encoded_array = ohe.fit_transform(train_data[existing_cols])  # 编码结果为二维数组（每行是一个向量）
    
#     # 保存原始类别和编码向量的映射关系
#     category_vector_mappings = {}
#     for i, col in enumerate(existing_cols):
#         # 获取该列的所有唯一类别
#         categories = ohe.categories_[i]
#         # 生成每个类别的One-Hot向量（如第0个类别对应[1,0,0,...]）
#         vectors = [[1 if j == idx else 0 for j in range(len(categories))] for idx in range(len(categories))]
#         # 存储 类别:向量 的映射
#         category_vector_mappings[col] = {cat: vec for cat, vec in zip(categories, vectors)}
    
#     # 打印特征映射关系（以向量形式展示）
#     print("\n特征映射关系（向量形式）:")
#     for col, mappings in category_vector_mappings.items():
#         print(f"\n{col} 的One-Hot向量映射:")
#         for cat, vec in mappings.items():
#             print(f"  {cat}: {vec}")
    
#     # 将编码后的数组转换为"向量字符串"列，保留原始列名（而非拆分多列）
#     for col_idx, col in enumerate(existing_cols):
#         # 提取该列对应的编码向量（从整体编码数组中切片）
#         # 注意：One-Hot编码后的数据是所有列的拼接，需定位当前列的向量位置
#         start = sum(len(ohe.categories_[i]) for i in range(col_idx))
#         end = start + len(ohe.categories_[col_idx])
#         # 将每个样本的向量转换为字符串（如"[1,0,0]"），便于存储和展示
#         train_data[f"{col}_onehot_vector"] = [str(encoded_array[row, start:end].tolist()) for row in range(len(train_data))]
#         # 删除原始分类列
#         train_data = train_data.drop(columns=[col])
    
#     # 打印处理后的数据示例（展示向量形式）
#     print("\nOne-Hot Encoding 处理后数据示例（向量形式）:")
#     vector_cols = [f"{col}_onehot_vector" for col in existing_cols]
#     print(train_data[vector_cols].head())
    
#     return train_data
def process_categorical_columns(train_data):
    """
    对分类变量进行One-Hot编码，删除原始分类列，用编码后的数值列替换
    输出格式可直接用于机器学习模型
    """
    categorical_cols = ['home_ownership', 'verification_status', 'purpose']
    existing_cols = check_columns_exist(train_data, categorical_cols)
    
    if not existing_cols:
        print("没有需要处理的分类列")
        return train_data
    
    # 执行One-Hot编码（生成数值型特征列）
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_array = ohe.fit_transform(train_data[existing_cols])  # 形状为 (样本数, 总类别数)
    
    # 获取编码后的列名（格式：原列名_类别值）
    encoded_col_names = ohe.get_feature_names_out(existing_cols)
    
    # 将编码结果转换为DataFrame
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_col_names, index=train_data.index)
    
    # 打印特征映射关系（展示原始类别与新列的对应关系）
    print("\n特征映射关系（原始类别 → 编码列）:")
    for i, col in enumerate(existing_cols):
        # 获取该列的所有原始类别
        categories = ohe.categories_[i]
        # 对应的编码列名
        start_idx = sum(len(ohe.categories_[j]) for j in range(i))
        end_idx = start_idx + len(categories)
        cols_for_category = encoded_col_names[start_idx:end_idx]
        # 打印映射
        print(f"\n{col}:")
        for cat, col_name in zip(categories, cols_for_category):
            print(f"  {cat} → {col_name} ")
    
    # 替换原分类列：删除原始列，拼接编码后的列
    train_data = train_data.drop(columns=existing_cols)  # 删除原始分类列
    train_data = pd.concat([train_data, encoded_df], axis=1)  # 拼接编码后的数值列
    
    # 打印处理后的数据示例（数值特征格式）
    print("\nOne-Hot Encoding 处理后数据示例（可直接用于模型）:")
    print(train_data[encoded_col_names].head())
    
    return train_data
# Label encoding 处理分类变量
def process_grade(train_data):
    if 'grade' in train_data.columns:
        grade_values = sorted(train_data['grade'].dropna().unique())
        grade_map = {grade: i for i, grade in enumerate(grade_values)}
        train_data['grade'] = train_data['grade'].map(grade_map).fillna(-1).astype(int)
        print(f"\ngrade 标签编码映射: {grade_map}")
        print("grade 编码后示例:", train_data['grade'].head())
    
    if 'sub_grade' in train_data.columns:
        sub_grade_values = sorted(train_data['sub_grade'].dropna().unique())
        sub_grade_map = {sub_grade: i for i, sub_grade in enumerate(sub_grade_values)}
        train_data['sub_grade'] = train_data['sub_grade'].map(sub_grade_map).fillna(-1).astype(int)
        print(f"\nsub_grade 标签编码映射: {sub_grade_map}")
        print("sub_grade 编码后示例:", train_data['sub_grade'].head())
    return train_data

def process_clustering_encoding(train_data):
    # 处理 emp_title 列
    if 'emp_title' in train_data.columns:
        train_data['emp_title'] = train_data['emp_title'].fillna('unknown')
        vectorizer_emp = TfidfVectorizer(max_features=1000, stop_words='english')
        emp_title_tfidf = vectorizer_emp.fit_transform(train_data['emp_title'])
        
        kmeans_emp = KMeans(n_clusters=30, random_state=42)
        train_data['emp_title_cluster'] = kmeans_emp.fit_predict(emp_title_tfidf)
        
        print("\nemp_title 聚类编码结果:")
        for cluster in range(min(5, kmeans_emp.n_clusters)):
            samples = train_data[train_data['emp_title_cluster'] == cluster]['emp_title'].sample(
                min(3, len(train_data[train_data['emp_title_cluster'] == cluster])), random_state=42)
            print(f"聚类 {cluster} 示例: {list(samples)}")
        
        emp_cluster_freq = train_data['emp_title_cluster'].value_counts(normalize=True).to_dict()
        train_data['emp_title_cluster_freq'] = train_data['emp_title_cluster'].map(emp_cluster_freq)
        train_data = train_data.drop('emp_title', axis=1)
    
    # 处理 title 列
    if 'title' in train_data.columns:
        train_data['title'] = train_data['title'].fillna('unknown')
        vectorizer_title = TfidfVectorizer(max_features=500, stop_words='english')
        title_tfidf = vectorizer_title.fit_transform(train_data['title'])
        
        kmeans_title = KMeans(n_clusters=20, random_state=42)
        train_data['title_cluster'] = kmeans_title.fit_predict(title_tfidf)
        
        print("\ntitle 聚类编码结果:")
        for cluster in range(min(5, kmeans_title.n_clusters)):
            samples = train_data[train_data['title_cluster'] == cluster]['title'].sample(
                min(3, len(train_data[train_data['title_cluster'] == cluster])), random_state=42)
            print(f"聚类 {cluster} 示例: {list(samples)}")
        
        title_cluster_freq = train_data['title_cluster'].value_counts(normalize=True).to_dict()
        train_data['title_cluster_freq'] = train_data['title_cluster'].map(title_cluster_freq)
        train_data = train_data.drop('title', axis=1)
    return train_data


# 执行特征处理并打印结果
print("=== 开始特征工程处理 ===")

train_data = process_binary_columns(train_data)
print("\n=== Binary Encoding 处理完成 ===")

train_data = process_categorical_columns(train_data)
print("\n=== One-Hot Encoding 处理完成 ===")

train_data = process_grade(train_data)
print("\n=== Grade 标签编码处理完成 ===")

train_data = process_clustering_encoding(train_data)
print("\n=== 聚类编码处理完成 ===")

print("\n=== 特征工程全部完成 ===")
print("最终数据维度:", train_data.shape)

# 保存处理后数据的前200行
if len(train_data) > 200:
    sampled_data = train_data.head(200)
else:
    sampled_data = train_data

new_file_path = '/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/dataset/train_data_encoded_sample.csv'
sampled_data.to_csv(new_file_path, index=False)
print(f"\n特征工程后的数据前200行已保存至: {new_file_path}")

# 保存完整处理后的训练数据（供特征筛选使用）
full_processed_path = '/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/dataset/train_data_encoded.csv'
train_data.to_csv(full_processed_path, index=False)
print(f"完整处理后的训练数据已保存至: {full_processed_path}")
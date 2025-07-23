import pandas as pd
from sklearn.model_selection import train_test_split
from data_preprocessing import (
    process_date_columns, process_numerical_columns,
    process_median_columns, process_categorical_columns, process_mean_columns,
    process_mode_columns, process_emp_length
)
from feature_engineering import (
    process_binary_columns, process_categorical_columns as process_onehot_columns,
    process_grade, process_clustering_encoding
)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

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

def create_data_pipeline(file_path, test_size=0.3, random_state=42):
    # 使用data_cleaning中的函数读取数据，保持一致性
    from data_cleaning import read_and_classify_data
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

    # 新增：保存划分后的原始训练集和测试集（未预处理）
    train_data = pd.concat([X_train_raw, y_train], axis=1)
    test_data = pd.concat([X_test_raw, y_test], axis=1)
    train_data.to_csv('ML/dataset/train_data.csv', index=False, encoding='utf-8-sig')
    test_data.to_csv('ML/dataset/test_data.csv', index=False, encoding='utf-8-sig')
    print("管道划分的原始训练集/测试集已保存为 CSV")

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
    

    # save processed data
    train_processed = pd.concat([X_train_processed, y_train], axis=1)
    test_processed = pd.concat([X_test_processed, y_test], axis=1)
    train_processed.to_csv('ML/dataset/train_data_encoded.csv', index=False)
    test_processed.to_csv('ML/dataset/test_data_encoded.csv', index=False)
    print("管道处理后的训练集/测试集已保存为 CSV")

    return X_train_processed, X_test_processed, y_train, y_test

# 添加执行入口
if __name__ == "__main__":
    # 设置数据文件路径
    file_path = '/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/dataset/cleaned_loan_data.csv'
    
    # 调用管道函数
    X_train, X_test, y_train, y_test = create_data_pipeline(
        file_path=file_path,
        test_size=0.3,
        random_state=42
    )
    
    print("数据处理管道执行完成！")
    print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
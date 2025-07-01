import pandas as pd
import numpy as np
import re

# 1.数据读取与分类模块
def read_and_classify_data(file_path):
    """读取数据并按类型分类"""
    # 读取原始数据
    data = pd.read_csv(file_path, low_memory=False)
    print(f"成功读取数据，行数: {data.shape[0]}, 列数: {data.shape[1]}")
    
    # 初始化列类型分类字典
    columns_dict = {}
    for col in data.columns:
        if data[col].dtype == 'object':
            columns_dict[col] = 'categorical'
        elif data[col].dtype in ['int64', 'float64']:
            columns_dict[col] = 'numerical'
        else:
            columns_dict[col] = 'other'
    
    # 汇总分类结果
    type_dict = {}
    for col, col_type in columns_dict.items():
        type_dict.setdefault(col_type, []).append(col)
    
    # 生成分类结果报告
    results = []
    for col_type, cols in type_dict.items():
        for col in cols:
            dtype = data[col].dtype
            nunique = data[col].nunique()
            missing = data[col].isna().sum()
            results.append([col_type, col, dtype, nunique, missing])
    
    # 保存分类结果
    results_df = pd.DataFrame(results, columns=['分类', '列名', '数据类型', '唯一值数量', '缺失值数量'])
    results_df.to_csv('分类结果.csv', index=False, encoding='utf-8-sig')
    print("数据分类完成，结果已保存至'分类结果.csv'")
    
    return data


# 2.数据预处理通用工具函数
def check_columns_exist(data, columns):
    """检查列是否存在并返回存在的列列表"""
    return [col for col in columns if col in data.columns]


def handle_missing_values(data, columns, fill_value):
    """用指定值填充缺失值"""
    if not columns:
        print("警告: 没有找到需要处理的列")
        return data
    data[columns] = data[columns].fillna(fill_value)
    return data


# 3.专项数据处理模块
#处理日期数据
def process_date_columns(data):
  # 定义日期列列表
    date_cols = ['issue_d', 'last_pymnt_d', 'earliest_cr_line', 'last_credit_pull_d', 'next_pymnt_d']
    existing_cols = check_columns_exist(data, date_cols)
      # 1. 处理issue_d, last_pymnt_d, earliest_cr_line, last_credit_pull_d
    for col in ['issue_d', 'last_pymnt_d', 'earliest_cr_line', 'last_credit_pull_d']:
        if col not in existing_cols:
            continue
            
        # 转换为datetime类型
        data[col] = pd.to_datetime(data[col], format='%b-%Y', errors='coerce')
        
        # 提取时间分量特征
        data[f"{col}_year"] = data[col].dt.year
        data[f"{col}_month"] = data[col].dt.month
        
        # 业务逻辑填充缺失值
        if col == 'earliest_cr_line' or col == 'last_pymnt_d':
            # 使用issue_d填充缺失值
            data[col] = data[col].fillna(data['issue_d'])
        elif col == 'last_credit_pull_d':
            # 最后一次信用查询日期缺失时，使用issue_d填充（业务逻辑：信用查询通常在贷款发放前）
            data[col] = data[col].fillna(data['issue_d'])
        # issue_d不填充，保持原始值
        
        # 处理填充后可能的NaT（理论上不应出现）
        data[f"{col}_year"] = data[f"{col}_year"].fillna(0).astype(int)
        data[f"{col}_month"] = data[f"{col}_month"].fillna(0).astype(int)
    
    # 2. 处理next_pymnt_d
    if 'next_pymnt_d' in existing_cols:
        # 转换为是否有日期的二分类特征
        data['next_pymnt_d_has_date'] = data['next_pymnt_d'].notna().astype(int)
    
    # 3. 删除原始日期列
    data = data.drop(columns=[col for col in existing_cols if col in data.columns])
    
    print("日期特征处理完成，已提取时间分量特征并删除原始列")
    return data

def process_numerical_columns(data):
    """处理数值型列,缺失值填充为0"""
    num_cols = [
        'mths_since_recent_bc_dlq', 'mths_since_last_major_derog', 'mths_since_recent_revol_delinq',
        'mths_since_last_delinq', 'mths_since_rcnt_il', 'mths_since_recent_inq', 'mths_since_recent_bc',
        'acc_open_past_24mths', 'collections_12_mths_ex_med', 'chargeoff_within_12_mths', 'all_util',
        'open_acc_6m', 'open_act_il', 'open_il_12m', 'open_il_24m', 'total_bal_il', 'open_rv_12m',
        'open_rv_24m', 'max_bal_bc', 'inq_fi', 'total_cu_tl', 'inq_last_12m', 'num_tl_120dpd_2m','percent_bc_gt_75',
        'il_util','tot_coll_amt','mo_sin_rcnt_tl','num_accts_ever_120_pd','num_actv_bc_tl','num_actv_rev_tl',
        'num_bc_tl','num_il_tl','num_op_rev_tl','num_rev_tl_bal_gt_0','num_tl_30dpd','num_tl_90g_dpd_24m',
        'num_tl_op_past_12m','num_bc_sats','num_sats','mort_acc','pub_rec_bankruptcies'
    ]
    existing_cols = check_columns_exist(data, num_cols)
    return handle_missing_values(data, existing_cols, 0)


def process_median_columns(data):
    """处理需要填充中位数的列"""
    median_cols = ['mo_sin_old_il_acct', 'percent_bc_gt_75', 'pct_tl_nvr_dlq', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op',
                   'num_rev_accts', 'revol_util', 'dti']
    existing_cols = check_columns_exist(data, median_cols)
    for col in existing_cols:
        median_val = data[col].median()
        data[col] = data[col].fillna(median_val)
    return data

def process_mean_columns(data):
    """处理需要填充均值的列"""
    mean_cols = ['bc_util','bc_open_to_buy','avg_cur_bal','tot_cur_bal','total_rev_hi_lim','tot_hi_cred_lim',
                 'total_il_high_credit_limit','total_bal_ex_mort','total_bc_limit']
    existing_cols = check_columns_exist(data, mean_cols)
    for col in existing_cols:
        mean_val = data[col].mean()
        data[col] = data[col].fillna(mean_val)
    return data

def process_mode_columns(data):
    """处理需要填充众数的列"""
    mode_cols = ['title','tax_liens']
    existing_cols = check_columns_exist(data, mode_cols)
    for col in existing_cols:
        mode_val = data[col].mode().iloc[0] if not data[col].mode().empty else np.nan
        data[col] = data[col].fillna(mode_val)
    return data


def process_categorical_columns(data):
    """处理分类数据列，缺失值填充为'unknown'"""
    cat_cols = ['emp_title']
    existing_cols = check_columns_exist(data, cat_cols)
    return handle_missing_values(data, existing_cols, 'unknown')

def process_emp_length(data):
    """处理员工工龄列，转换为数值型"""
    if 'emp_length' not in data.columns:
        print("警告: 未找到'emp_length'列")
        return data
    
    emp_map = {
        "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
        "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9, "10+ years": 10
    }
    data['emp_length'] = data['emp_length'].map(emp_map).fillna(-1).astype(int)
    return data



# 4.主处理流程
def main():
    """数据处理主流程"""
    try:
        #1. 读取并分类数据
        file_path = '/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/cleaned_loan_data_v2.csv'
        data = read_and_classify_data(file_path)
                                      
        #2. 按顺序执行数据处理
        data = process_date_columns(data)
        data = process_numerical_columns(data)
        data = process_median_columns(data)
        data = process_categorical_columns(data)
        data = process_mean_columns(data)
        data = process_mode_columns(data)
        data = process_emp_length(data)
        
        #3. 保存处理后的数据
        output_path = '/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/processed_loan_data.csv'
        data.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"数据处理完成，结果已保存至: {output_path}")
        
        #4. 输出关键处理结果检查
        if 'emp_length' in data.columns:
            print("\nemployee length 列处理结果:")
            print(data['emp_length'].value_counts())

        #5. 检查所有列的缺失值情况
        print("\n缺失值填补检查:")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(data.isnull().sum())
        
    except Exception as e:
        print(f"数据处理失败，错误信息: {str(e)}")
        exit(1)


# 执行主程序
if __name__ == "__main__":
    main()


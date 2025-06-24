# 数据分类
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import re

# 读取数据
cleaned_data = pd.read_csv('/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/cleaned_loan_data_v2.csv', low_memory=False)

# 初始化分类字典
columns_dict = {}
for column in cleaned_data.columns:
    if cleaned_data[column].dtype == 'object':
        columns_dict[column] = 'categorical'
    elif cleaned_data[column].dtype in ['int64', 'float64']:
        columns_dict[column] = 'numerical'
    else:
        columns_dict[column] = 'other'

# 汇集相同col_type的列名
type_dict = {}
for column, col_type in columns_dict.items():
    if col_type not in type_dict:
        type_dict[col_type] = []
    type_dict[col_type].append(column)

# 创建一个空的列表来保存分类结果
results = []

# 打印分类结果并保存到列表
for col_type, columns in type_dict.items():
    for column in columns:
        dtype = cleaned_data[column].dtype
        nunique = cleaned_data[column].nunique()
        missing_values = cleaned_data[column].isna().sum()
        results.append([col_type, column, dtype, nunique, missing_values])

# 创建 DataFrame
results_df = pd.DataFrame(results, columns=['分类', '列名', '数据类型', '唯一值数量', '缺失值数量'])

# 保存到 CSV 文件
results_df.to_csv('分类结果.csv', index=False, encoding='utf-8-sig')

# 1. 处理日期型数据
def process_date_data(cleaned_data):
    # 确保列存在
    date_columns = ['last_pymnt_d', 'next_pymnt_d']
    existing_date_cols = [col for col in date_columns if col in cleaned_data.columns]
    
    for col in existing_date_cols:
        cleaned_data[col] = pd.to_datetime(cleaned_data[col], format='%b-%Y', errors='coerce')
        # 转换为Unix时间戳(秒)
        cleaned_data[col] = cleaned_data[col].astype('int64') // 10**9
    
    return cleaned_data

# 2. 处理数值型数据
def process_numerical_data(cleaned_data):
    # 定义要处理的列，并检查是否存在于数据集中
    columns_to_fill = ['mths_since_recent_bc_dlq', 'mths_since_last_major_derog', 'mths_since_recent_revol_delinq','mths_since_last_delinq','mths_since_rcnt_il',
                       'mths_since_recent_inq','mths_since_recent_bc','acc_open_past_24mths','collections_12_mths_ex_med','chargeoff_within_12_mths']
    existing_columns = [col for col in columns_to_fill if col in cleaned_data.columns]
    
    if not existing_columns:
        print("警告: 没有找到需要处理的列")
        return cleaned_data
    
    # 填补缺失值为0
    cleaned_data[existing_columns] = cleaned_data[existing_columns].fillna(0)
    
    return cleaned_data

# 执行数据处理
try:
    cleaned_data = process_date_data(cleaned_data)
    cleaned_data = process_numerical_data(cleaned_data)
except Exception as e:
    print(f"数据处理过程中出错: {e}")
    exit(1)

# 检查缺失值是否已填补
print("\n缺失值填补检查:")
check_columns = ['mths_since_recent_bc_dlq', 'mths_since_last_major_derog', 'mths_since_recent_revol_delinq','mths_since_last_delinq','mths_since_rcnt_il',
                       'mths_since_recent_inq','mths_since_recent_bc','acc_open_past_24mths','collections_12_mths_ex_med','chargeoff_within_12_mths']
existing_check_cols = [col for col in check_columns if col in cleaned_data.columns]
print(cleaned_data[existing_check_cols].isnull().sum())


# #对所有的列根据研究目的重新分类
# def categorize_columns(df):
#     # 定义分类字典
#     columns_dict = {
#         '贷款核心属性类(贷款基本条款)': ['id','loan_amnt','funded_amnt','funded_amnt_inv','term','int_rate',
#                             'installment','grade', 'sub_grade', 'initial_list_status','issue_d','initial_list_status',
#                             'disbursement_method'],
#         '借款人信息类(个人背景特征)': ['emp_title','emp_length','home_ownership','annual_inc','addr_state', 
#                            'application_type', 'verification_status'],
#         '信用评估类(核心风险指标)': ['fico_range_low', 'fico_range_high', 'last_fico_range_high', 'last_fico_range_low',
#                            'pub_rec', 'pub_rec_bankruptcies', 'tax_liens','earliest_cr_line','last_credit_pull_d',
#                            'inq_last_6mths','inq_fi', 'inq_last_12m', 'pub_rec', 'total_cu_tl', 'all_util', 
#                            'il_util', 'bc_util', 'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'tot_hi_cred_lim', 
#                            'total_bc_limit', 'total_il_high_credit_limit'],
#         '还款能力类':['annual_inc', 'dti', 'tot_coll_amt', 'tot_cur_bal', 'total_bal_ex_mort', 'total_rev_hi_lim',
#                  'payment_plan','revol_bal','revol_util','avg_cur_bal', 'bc_open_to_buy', 'total_bal_il', 'tot_cur_bal'],
#         '违约强相关类(贷款违约信息)': ['loan_status', 'delinq_2yrs','mths_since_last_delinq', 'acc_now_delinq', 
#                            'delinq_amnt', 'num_accts_ever_120_pd', 'num_tl_120dpd_2m', 'num_tl_30dpd', 
#                            'num_tl_90g_dpd_24m', 'debt_settlement_flag','hardship_flag','recoveries',
#                            'collection_recovery_fee', 'collections_12_mths_ex_med', 'mths_since_last_major_derog', 
#                            'chargeoff_within_12_mths', 'mths_since_recent_bc_dlq', 'mths_since_recent_revol_delinq'],
#         '还款活动类':['last_pymnt_d','next_pymnt_d','out_prncp','out_prncp_inv','total_pymnt','total_pymnt_inv',
#                  'last_pymnt_amnt', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee'],
#         '账户活动类':['open_acc', 'total_acc', 'open_acc_6m', 'open_act_il', 'open_il_12m', 'open_il_24m', 'open_rv_12m',
#                   'open_rv_24m', 'mths_since_rcnt_il', 'total_bal_il', 'max_bal_bc', 'open_rv_12m', 'open_rv_24m', 
#                   'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl', 
#                   'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_op_past_12m', 'acc_open_past_24mths', 
#                   'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc',
#                   'mths_since_recent_bc', 'mths_since_recent_inq', 'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
#                   ],
#         '其他相关类':['title','zip_code','policy_code'],


#     }
#     return columns_dict


# # 贷款核心属性类(贷款基本条款)缺失值检查
# def check_core_loan_attributes(df):
#     # 获取分类字典
#     columns_dict = categorize_columns(df)
    
#     # 提取贷款核心属性类的列
#     core_loan_attributes = columns_dict['贷款核心属性类(贷款基本条款)']
    
#     # 检查列是否存在于数据集中
#     existing_columns = [col for col in core_loan_attributes if col in df.columns]
#     if len(existing_columns) != len(core_loan_attributes):
#         missing_cols = [col for col in core_loan_attributes if col not in df.columns]
#         print(f"警告: 以下列不在数据集中: {missing_cols}")
    
#     # 统计缺失值
#     missing_values = df[existing_columns].isna().sum()
#     print(missing_values)

# check_core_loan_attributes
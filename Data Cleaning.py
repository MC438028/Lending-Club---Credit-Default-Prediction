# 数据清洗
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import re

# 读取CSV文件
loan = pd.read_csv('/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/accepted_2007_to_2018Q4.csv')
loan.info()
headers = loan.columns.tolist()
print(loan.shape)
#print(headers)

# 统计贷款申请类型
print(loan['application_type'].value_counts(dropna=False))

#缺失值检测与统计
missing_stats = loan.isna().sum()
total_rows = len(loan)
print(total_rows)
missing_ratio = (missing_stats / total_rows).round(4)
missing_cols = missing_stats[missing_stats > 0].sort_values(ascending=False)

missing_df = pd.DataFrame({
    'Headers': missing_cols.index,
    'Missing Values': missing_cols.values,
    'Missing Ratio': missing_ratio[missing_cols.index]
})

# 保存到CSV文件
csv_path = '/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/missing_values.csv'
missing_df.to_csv(csv_path, index=False, encoding='utf-8')
print(f"\n缺失值统计结果已保存至: {csv_path}")

# 定义低价值列列表
low_value_columns = [
    # 文本描述类（对违约预测帮助小）
     'desc', 'url','zip_code', 'addr_state'
    # 特殊还款计划相关（若目标为常规违约预测）
    'hardship_flag', 'hardship_type', 'hardship_reason', 'hardship_status',
    'deferral_term', 'hardship_amount', 'hardship_start_date', 'hardship_end_date',
    'payment_plan_start_date', 'hardship_length', 'hardship_dpd', 'hardship_loan_status',
    'orig_projected_additional_accrued_interest', 'hardship_payoff_balance_amount',
    'hardship_last_payment_amount', 'debt_settlement_flag', 'debt_settlement_flag_date',
    'settlement_status', 'settlement_date', 'settlement_amount', 'settlement_percentage',
    'settlement_term',
    # 次级申请人信息/联合申请相关特征
    'sec_app_fico_range_low', 'sec_app_fico_range_high', 'sec_app_earliest_cr_line',
    'sec_app_inq_last_6mths', 'sec_app_mort_acc', 'sec_app_open_acc', 'sec_app_revol_util',
    'sec_app_open_act_il', 'sec_app_num_rev_accts', 'sec_app_chargeoff_within_12_mths',
    'sec_app_collections_12_mths_ex_med', 'sec_app_mths_since_last_major_derog', 'revol_bal_joint',
    'verfication_status_joint'
]

# 分阶段删除策略
deleted_columns = []

# 1. 删除全空列
all_empty_cols = missing_stats[missing_stats == total_rows].index
if len(all_empty_cols) > 0:
    loan = loan.drop(columns=all_empty_cols)
    deleted_columns.extend(all_empty_cols)
    print(f"阶段1: 删除{len(all_empty_cols)}个全空列")

# 2. 删除低价值且高缺失率列（缺失率>50%）
high_missing_low_value = [col for col in low_value_columns 
                          if col in loan.columns and missing_ratio[col] > 0.5]
if len(high_missing_low_value) > 0:
    loan = loan.drop(columns=high_missing_low_value)
    deleted_columns.extend(high_missing_low_value)
    print(f"阶段2: 删除{len(high_missing_low_value)}个低价值高缺失列")

# 3. 删除高缺失率且业务无关列（缺失率>80%且不在保留列表中）
key_columns = [
    'loan_status', 'loan_amnt', 'funded_amnt', 'int_rate', 'installment', 
    'grade', 'sub_grade', 'emp_length', 'home_ownership', 'annual_inc',
    'verification_status', 'issue_d', 'purpose', 'dti', 'delinq_2yrs',
    'earliest_cr_line', 'fico_range_low', 'fico_range_high', 'inq_last_6mths',
    'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
    'total_pymnt', 'total_rec_prncp', 'total_rec_int', 'last_credit_pull_d',
    'collections_12_mths_ex_med', 'mths_since_last_major_derog',
    'application_type', 'tot_coll_amt', 'tot_cur_bal'
]  # 关键列列表，可根据业务调整

high_missing_irrelevant = [col for col in loan.columns 
                          if col not in key_columns 
                          and missing_ratio[col] > 0.8 
                          and col not in deleted_columns]
if len(high_missing_irrelevant) > 0:
    loan = loan.drop(columns=high_missing_irrelevant)
    deleted_columns.extend(high_missing_irrelevant)
    print(f"阶段3: 删除{len(high_missing_irrelevant)}个高缺失无关列")

# 4. 删除低价值列（不在保留列表中）
low_value_irrelevant = [col for col in low_value_columns 
                        if col in loan.columns and col not in deleted_columns]
if len(low_value_irrelevant) > 0:
    loan = loan.drop(columns=low_value_irrelevant)
    deleted_columns.extend(low_value_irrelevant)
    print(f"阶段4: 删除{len(low_value_irrelevant)}个低价值列")


# 5. 单独删除指定列（无论缺失率如何）
specific_columns_to_drop = ['addr_state','hardship_flag'] 
columns_to_delete = [col for col in specific_columns_to_drop 
                     if col in loan.columns and col not in deleted_columns]

if columns_to_delete:
    loan = loan.drop(columns=columns_to_delete)
    deleted_columns.extend(columns_to_delete)
    print(f"阶段5: 单独删除{len(columns_to_delete)}个指定列: {columns_to_delete}")

# 输出删除的列及最终数据形状
print(f"\n共删除{len(deleted_columns)}列，删除的列名:\n{deleted_columns}")
print(f"数据清洗后形状: {loan.shape}")

# 保存清洗后数据
cleaned_csv_path = '/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/cleaned_loan_data.csv'
loan.to_csv(cleaned_csv_path, index=False, encoding='utf-8')
print(f"\n清洗后数据已保存至: {cleaned_csv_path}")

new_data= pd.read_csv('/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/cleaned_loan_data.csv')
print(new_data.tail(10))
# 筛选出缺失率为0的列（即理论上不应有缺失值的列）
missing_ratio_after_step1 = (new_data.isna().sum() / len(new_data)).round(4)
zero_missing_ratio_cols = missing_ratio_after_step1[missing_ratio_after_step1 == 0].index.tolist()
print(f"\n缺失率为0的列: {zero_missing_ratio_cols}")

# 检查这些列中是否存在实际缺失值
actual_missing_in_zero_cols = new_data[zero_missing_ratio_cols].isna().sum()
print(f"\n{zero_missing_ratio_cols}列中实际缺失值数量:")
print(actual_missing_in_zero_cols[actual_missing_in_zero_cols > 0])

# 删除缺失率为0的列中存在缺失值的行
original_rows = new_data.shape[0]
rows_dropped = new_data.dropna(subset=zero_missing_ratio_cols)
remaining_rows = rows_dropped.shape[0]
num_of_rows_dropped = original_rows - remaining_rows

# 同时删除其他关键列的缺失值行（根据业务需求添加）
key_cols_to_drop = ['annual_inc', 'inq_last_6mths', 'last_credit_pull_d']
rows_dropped = rows_dropped.dropna(subset=key_cols_to_drop)
additional_dropped = remaining_rows - rows_dropped.shape[0]
num_of_rows_dropped += additional_dropped

# 打印结果
print(f"\n共删除 {num_of_rows_dropped} 行")
print(f"深度清洗后数据形状: {rows_dropped.shape}")

# 检查清洗后数据的缺失值情况
final_missing_stats = rows_dropped.isna().sum()
print(f"\n深度清洗后各列缺失值数量:")
print(final_missing_stats[final_missing_stats > 0])

# 保存深度清洗后数据
cleaned_csv_path = '/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/cleaned_loan_data_v2.csv'
rows_dropped.to_csv(cleaned_csv_path, index=False, encoding='utf-8')
print(f"\n深度清洗后数据已保存至: {cleaned_csv_path}")
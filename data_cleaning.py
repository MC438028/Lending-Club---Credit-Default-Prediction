# 数据清洗
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import re

# 读取CSV文件
loan = pd.read_csv('/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/dataset/accepted_2007_to_2018Q4.csv')
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

# 定义关键配置（根据features_review.xlsx中relevant to default=0的特征整理）
# 1. 所有relevant to default=0的特征（需删除的违约无关列）
default_irrelevant_cols = [
    # irrelevent
    'addr_state', 'desc','id', 'last_credit_pull_d', 'member_id', 'policy_code', 
    'purpose', 'title', 'url', 'zip_code',
    # only record date-irrelevent 
    'debt_settlement_flag_date', 'issue_d', 'payment_plan_start_date', 'settlement_date',
    # hardship-plan
    'deferral_term', 'hardship_amount', 'hardship_dpd', 'hardship_end_date', 'hardship_last_payment_amount', 
    'hardship_length', 'hardship_loan_status', 'hardship_payoff_balance_amount', 'hardship_reason', 
    'hardship_start_date', 'hardship_status', 'hardship_type', 'orig_projected_additional_accrued_interest',
    # co-borrower
    'annual_inc_joint', 'dti_joint', 'revol_bal_joint', 'sec_app_chargeoff_within_12_mths',
    'sec_app_collections_12_mths_ex_med', 'sec_app_earliest_cr_line', 'sec_app_fico_range_high',
    'sec_app_fico_range_low', 'sec_app_inq_last_6mths', 'sec_app_mort_acc', 'sec_app_mths_since_last_major_derog',
    'sec_app_num_rev_accts', 'sec_app_open_acc', 'sec_app_open_act_il', 'sec_app_revol_util', 'verification_status_joint',
    # other irrelevant
    'chargeoff_within_12_mths', 'collection_recovery_fee', 'collections_12_mths_ex_med', 'debt_settlement_flag', 
    'funded_amnt_inv','grade', 'last_fico_range_high', 'last_fico_range_low', 'last_pymnt_amnt', 'last_pymnt_d', 
    'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'pymnt_plan', 'recoveries', 'settlement_amount', 
    'settlement_percentage', 'settlement_status', 'settlement_term', 'total_pymnt', 'total_pymnt_inv', 
    'total_rec_int', 'total_rec_late_fee', 'total_rec_prncp', 'total_rev_hi_lim'
]
# 2. 从违约无关列中筛选note含post-loan前缀的特征（需单独保存）
post_loan_cols = [
    'chargeoff_within_12_mths', 'collection_recovery_fee', 'collections_12_mths_ex_med', 'debt_settlement_flag',
    'last_fico_range_high', 'last_fico_range_low', 'last_pymnt_amnt', 'last_pymnt_d', 'next_pymnt_d', 
    'out_prncp', 'out_prncp_inv', 'pymnt_plan', 'recoveries', 'settlement_amount', 'settlement_percentage', 
    'settlement_status', 'settlement_term', 'total_pymnt', 'total_pymnt_inv', 'total_rec_int',
      'total_rec_late_fee', 'total_rec_prncp', 'total_rev_hi_lim'
]  

# 数据清洗步骤
deleted_columns = []

# 第一步：删除全空列
all_empty_cols = missing_stats[missing_stats == total_rows].index.tolist()
if all_empty_cols:
    loan = loan.drop(columns=all_empty_cols)
    deleted_columns.extend(all_empty_cols)
    print(f"阶段1: 删除{len(all_empty_cols)}个全空列")

# 第二步：删除高缺失值列（缺失率>80%）
high_missing_irrelevant = [
    col for col in loan.columns if missing_ratio[col] > 0.8
]
if high_missing_irrelevant:
    loan = loan.drop(columns=high_missing_irrelevant)
    deleted_columns.extend(high_missing_irrelevant)
    print(f"阶段2: 删除{len(high_missing_irrelevant)}个高缺失值列")

# 第三步：删除剩余所有违约无关列（relevant=0）
remaining_irrelevant = [
    col for col in default_irrelevant_cols
    if col in loan.columns and col not in deleted_columns
]
if remaining_irrelevant:
    # 关键修复：只保留当前loan中存在的post_loan列
    existing_post_loan_cols = [col for col in post_loan_cols if col in loan.columns]
    if existing_post_loan_cols:  # 仅当有存在的列时才保存
        post_loan_data = loan[existing_post_loan_cols].copy()
        post_loan_path = '/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/post_loan_features.csv'
        post_loan_data.to_csv(post_loan_path, index=False, encoding='utf-8')
        print(f"已保存post-loan特征至: {post_loan_path}（共{len(existing_post_loan_cols)}列）")
    else:
        print("无可用的post-loan特征（所有post_loan列已被删除），未保存文件")
    
    # 再删除所有relevant=0的列
    loan = loan.drop(columns=remaining_irrelevant)
    deleted_columns.extend(remaining_irrelevant)
    print(f"阶段3: 删除{len(remaining_irrelevant)}个剩余违约无关列")



# 输出删除的列及最终数据形状
print(f"\n共删除{len(deleted_columns)}列，删除的列名:\n{deleted_columns}")
print(f"数据清洗后形状: {loan.shape}")

# 第四步：处理行级缺失值
# 初始化总删除行数计数器
total_rows_dropped = 0
# 4.1 筛选理论上不应有缺失值的列（缺失率=0）
# 新增：检查理论上零缺失列的实际缺失值
missing_ratio_after = (loan.isna().sum() / len(loan)).round(4)
zero_missing_cols = missing_ratio_after[missing_ratio_after == 0].index.tolist()

# 检查这些列中是否存在实际缺失值
actual_missing_in_zero_cols = loan[zero_missing_cols].isna().sum()
print(f"\n===== 零缺失列实际缺失值检查 =====")
print(f"理论上缺失率为0的列: {zero_missing_cols}")
print(f"这些列中实际存在的缺失值数量:")
print(actual_missing_in_zero_cols[actual_missing_in_zero_cols > 0])  # 只显示有缺失的列

# 4.2 删除这些列中存在缺失值的行
original_rows = loan.shape[0]
loan = loan.dropna(subset=zero_missing_cols)
rows_dropped_zero = original_rows - loan.shape[0]
total_rows_dropped += rows_dropped_zero 
print(f"阶段4.1: 删除{rows_dropped_zero}行（零缺失列存在缺失值）")

# 4.3 删除关键列的缺失值行（根据业务补充）
key_cols = ['annual_inc', 'inq_last_6mths', 'fico_range_low']  # 示例关键列
original_rows = loan.shape[0]
loan = loan.dropna(subset=key_cols)
rows_dropped_key = original_rows - loan.shape[0]
total_rows_dropped += rows_dropped_key
print(f"阶段4.2: 删除{rows_dropped_key}行（关键列存在缺失值）")

# 4.4 新增：删除特定条件行
# 删除'loan_status'为Late (31-120 days)且'hardship_flag'为'N'的行
original_rows = loan.shape[0]
loan = loan[~((loan['loan_status'] == 'Late (31-120 days)') & (loan['hardship_flag'] == 'Y'))]
rows_dropped_late = original_rows - loan.shape[0]
total_rows_dropped += rows_dropped_late
print(f"阶段4.3: 删除{rows_dropped_late}行（loan_status=Late (31-120 days)且hardship_flag=N）")

# 第五步：删除指定列
cols_to_drop = ['hardship_flag', 'application_type']
if cols_to_drop:
    loan = loan.drop(columns=cols_to_drop)
    deleted_columns.extend(cols_to_drop)
    print(f"阶段5: 删除{len(cols_to_drop)}列（hardship_flag和application_type）")

# 新增：最终缺失值检查（保存前确认）
final_missing = loan.isna().sum()
final_missing_cols = final_missing[final_missing > 0]

print("\n===== 最终缺失值检查 =====")
if final_missing_cols.empty:
    print("所有列均无缺失值，数据完整性符合要求。")
else:
    print("存在缺失值的列及数量：")
    for col, count in final_missing_cols.items():
        print(f"{col}: {count}")

# 输出清洗结果
print(f"\n总删除列数: {len(deleted_columns)}")
print(f"总删除行数: {total_rows_dropped}")
print(f"清洗后数据形状: {loan.shape}")

# 保存最终清洗数据
final_cleaned_path = '/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/dataset/cleaned_loan_data.csv'
loan.to_csv(final_cleaned_path, index=False, encoding='utf-8')
print(f"最终清洗数据已保存至: {final_cleaned_path}")
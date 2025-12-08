"""
通用工具函数：数据加载、模型训练等
"""

import pandas as pd
import numpy as np
from pathlib import Path

class DataLoader:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        
    def load_data(self):   
        train_data = pd.read_csv(self.train_path)
        test_data = pd.read_csv(self.test_path)
        
        X_train = train_data.drop(columns=['loan_status_flag'])
        y_train = train_data['loan_status_flag']
        X_test = test_data.drop(columns=['loan_status_flag'])
        y_test = test_data['loan_status_flag']
        
        print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
        
        return X_train, y_train, X_test, y_test

# 全局配置
DATA_CONFIG = {
    'train_path': '/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/dataset/train_data_encoded.csv',
    'test_path': '/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/dataset/test_data_encoded.csv'
}

def get_data():
    loader = DataLoader(DATA_CONFIG['train_path'], DATA_CONFIG['test_path'])
    return loader.load_data()
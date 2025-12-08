"""
RQ1: How do distinct feature selection subsetologies impact predictive performance?
- 对9个特征选择子集进行5折交叉验证
- 保存每折的详细结果用于后续分析
"""

import pickle
import numpy as np
import pandas as pd
import json
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (confusion_matrix, roc_auc_score, classification_report,
                             roc_curve, auc, ConfusionMatrixDisplay )

from lightgbm import LGBMClassifier
from utils import get_data


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.feature_names]

class Step1CrossValidator:
    def __init__(self, feature_file, output_dir="results"):
        self.feature_file = feature_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 加载数据
        print("="*60)
        print("加载数据...")
        print("="*60)
        self.X_train, self.y_train, self.X_test, self.y_test = get_data()

        # 加载特征子集
        print("\n加载特征子集...")
        with open(self.feature_file, "rb") as f:
            self.feature_dict = pickle.load(f)
        
        print(f"加载了 {len(self.feature_dict)} 个特征选择方法:")
        for name, features in self.feature_dict.items():
            print(f"  - {name}: {len(features)} 个特征")

        # 存储结果
        self.cv_results_all = {}
        self.test_results_all = {}
        
        # LightGBM 参数
        self.lgb_base_params = {
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        # GridSearch 参数网格
        self.param_grid = {
            'model__n_estimators': [400, 500],
            'model__max_depth': [6],
            'model__learning_rate': [0.1],
        }
    
    def cross_validate_single_subset(self, subset_name, features, n_folds=5):
        # 对单个特征选择方法进行交叉验证
        print(f"\n{'='*60}")
        print(f"处理方法: {subset_name} ({len(features)} 个特征)")
        print(f"{'='*60}")
        
        cv_results = []
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(self.X_train, self.y_train)):          
            X_tr, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_tr, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            
            # 构建 Pipeline
            pipeline = Pipeline([
                ('feature_selector', FeatureSelector(features)),
                ('model', LGBMClassifier(**self.lgb_base_params))
            ])
            
            # GridSearchCV
            gs = GridSearchCV(
                pipeline,
                param_grid=self.param_grid,
                scoring='roc_auc',
                cv=3,
                n_jobs=-1,
                verbose=0,
                return_train_score=True
            )
            
            # 训练
            gs.fit(X_tr, y_tr)
            
            # 最佳模型预测
            best_model = gs.best_estimator_
            y_val_pred_proba = best_model.predict_proba(X_val)[:, 1]
            y_val_pred = (y_val_pred_proba > 0.5).astype(int)
            
            # confusion matrix
            cm_fold = confusion_matrix(y_val, y_val_pred)
            tn_fold, fp_fold, fn_fold, tp_fold = cm_fold.ravel()
            tpr_fold = tp_fold / (tp_fold + fn_fold) if (tp_fold + fn_fold) > 0 else 0.0
            fpr_fold = fp_fold / (fp_fold + tn_fold) if (fp_fold + tn_fold) > 0 else 0.0
            
            # AUC + RMSE
            fold_val_auc = roc_auc_score(y_val, y_val_pred_proba)
            fold_train_auc = roc_auc_score(y_tr, best_model.predict_proba(X_tr)[:, 1])
            rmse_fold = np.sqrt(np.mean((y_val - y_val_pred_proba) ** 2))

            # ROC 曲线
            fpr_curve, tpr_curve, _ = roc_curve(y_val, y_val_pred_proba)
            roc_auc_curve = auc(fpr_curve, tpr_curve)

            # 计算指标
            cv_results.append({
            'Subset': subset_name,
            'Fold_Index': fold_idx + 1,
            'Features': len(features),
            'Best_Params': str(gs.best_params_),
            'Fold_Train_ROC_AUC': fold_train_auc,
            'Fold_Val_ROC_AUC': fold_val_auc,
            'Fold_Val_Recall': tpr_fold,
            'Fold_Val_FPR': fpr_fold,
            'Fold_Val_RMSE': rmse_fold,
            'y_val': y_val.tolist(),
            'y_val_pred_proba': y_val_pred_proba.tolist(),
            'Fold_FPR': fpr_curve.tolist(),
            'Fold_TPR': tpr_curve.tolist(),
            'Fold_ROC_AUC': roc_auc_curve
    })

            print(f"  Fold {fold_idx+1}: Val AUC={fold_val_auc:.4f}, Recall={tpr_fold:.4f}, FPR={fpr_fold:.4f}, RMSE={rmse_fold:.4f}")
        
        return cv_results
    
    def evaluate_on_test(self, subset_name, features):
        """在测试集上评估"""
        print(f"\n 测试集评估: {subset_name}")

        # 构建 Pipeline
        pipeline = Pipeline([
            ('feature_selector', FeatureSelector(features)),
            ('model', LGBMClassifier(**self.lgb_base_params))
        ])
        
        # GridSearchCV
        gs = GridSearchCV(
            pipeline,
            param_grid=self.param_grid,
            scoring='roc_auc',
            cv=3,
            n_jobs=-1,
            verbose=0
        )
        
        # 训练
        gs.fit(self.X_train, self.y_train)
        
        # 预测
        best_model = gs.best_estimator_
        y_pred_proba = best_model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        train_auc = roc_auc_score(self.y_train, best_model.predict_proba(self.X_train)[:, 1])
        test_auc = roc_auc_score(self.y_test, y_pred_proba)
        rmse = np.sqrt(np.mean((self.y_test - y_pred_proba) ** 2))

        fpr_test, tpr_test, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc_test = auc(fpr_test, tpr_test)

                
        test_results = {
            'Subset': subset_name,
            'Best_Params': str(gs.best_params_),
            'Features': len(features),
            'Train_ROC_AUC': train_auc,
            'Test_ROC_AUC': test_auc,
            'Recall': tpr,
            'FPR': fpr,
            'RMSE': rmse,
            'y_test': self.y_test.tolist(),                  
            'y_test_pred_proba': y_pred_proba.tolist(),     
            'tpr_test': tpr_test.tolist(),
            'roc_auc_test': roc_auc_test
                }

        print(f"    测试集 ROC-AUC: {test_results['Test_ROC_AUC']:.4f}")
        print(f"    Best params: {test_results['Best_Params']}")

        return test_results

    def run_all_subsets(self):
        """运行所有特征选择方法的交叉验证和测试集评估"""
        print("\n" + "="*60)
        print("开始处理所有特征选择方法")
        print("="*60)
        
        for subset_name, features in self.feature_dict.items():
            # 交叉验证
            cv_result = self.cross_validate_single_subset(subset_name, features)
            self.cv_results_all[subset_name] = cv_result
            
            # 测试集评估
            test_result = self.evaluate_on_test(subset_name, features)
            self.test_results_all[subset_name] = test_result
        
        # 保存结果
        self._save_results()
        
        # 计算排名
        self._calculate_rankings()
    
    def _save_results(self):
        """保存结果到 CSV 和 PKL"""
        print("\n" + "="*60)
        print("保存结果...")
        print("="*60)
        
        # === 1. 交叉验证结果 ===
        cv_summary = []
        for subset, results in self.cv_results_all.items():
            fold_metrics_df = pd.DataFrame(results)
            mean_metrics = fold_metrics_df.mean(numeric_only=True)
            std_metrics = fold_metrics_df.std(numeric_only=True)
            
            cv_summary.append({
                'Subset': subset,
                'Mean_Train_ROC_AUC': mean_metrics['Fold_Train_ROC_AUC'],
                'Mean_Val_ROC_AUC': mean_metrics['Fold_Val_ROC_AUC'],
                'Std_Val_ROC_AUC': std_metrics['Fold_Val_ROC_AUC'],
                'Mean_Val_Recall': mean_metrics['Fold_Val_Recall'],
                'Mean_Val_FPR': mean_metrics['Fold_Val_FPR'],
                'Mean_Val_RMSE': mean_metrics['Fold_Val_RMSE'],
                'Features': fold_metrics_df['Features'].iloc[0]
            })
        
        self.cv_summary_df = pd.DataFrame(cv_summary)
        self.cv_summary_df.to_csv(self.output_dir / 'rq1_cv_metrics.csv', index=False)
        print(f"交叉验证结果已保存: {self.output_dir / 'rq1_cv_metrics.csv'}")
        
        # === 2. 测试集结果 ===
        test_df = pd.DataFrame(self.test_results_all.values())
        test_df.to_csv(self.output_dir / 'rq1_test_metrics.csv', index=False)
        print(f"✓ 测试集结果已保存: {self.output_dir / 'rq1_test_metrics.csv'}")
        
        
        # === 3. 保存完整结果 (PKL) - 包含预测值用于可视化 ===
        with open(self.output_dir / 'cv_results_detailed.pkl', 'wb') as f:
            pickle.dump({
                'cv_results': self.cv_results_all,
                'test_results': self.test_results_all,
            }, f)
        print(f"✓ 详细结果已保存: {self.output_dir / 'cv_results_detailed.pkl'}")
    
    def _calculate_rankings(self):
        """计算排名并选出 Top 4 方法"""
        print("\n" + "="*60)
        print("计算综合排名...")
        print("="*60)
        
        # 排序
        self.cv_summary_df = self.cv_summary_df.sort_values(
            'Mean_Val_ROC_AUC', ascending=False
        ).reset_index(drop=True)
        
        # Top 4
        self.top4_subsets = self.cv_summary_df.head(4)['Subset'].tolist()

        print("\n Top 4 特征选择方法:")
        print(self.cv_summary_df.head(4)[[
            'Subset', 'Features','Mean_Val_ROC_AUC', 'Mean_Val_Recall', 
            'Mean_Val_FPR' 
        ]].to_string(index=False))
        
        # 保存 Top 4
        with open(self.output_dir / 'top4_subsets.pkl', 'wb') as f:
            pickle.dump(self.top4_subsets, f)
        print(f"\n✓ Top 4 方法已保存: {self.output_dir / 'top4_subsets.pkl'}")

        # 保存排序后的完整排名
        self.cv_summary_df.to_csv(
            self.output_dir / 'rq1_cv_metrics_ranked.csv', index=False
        )


# ===== 主程序入口 =====
if __name__ == '__main__':
    print("\n" + "="*60)
    print("RQ1: 特征选择方法对比实验")
    print("="*60)
    
    validator = Step1CrossValidator(
        feature_file='/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/Project/data/selected_features.pkl',
        output_dir='results'
    )
    
    # 运行所有方法
    validator.run_all_subsets()
    
    print("\n" + "="*60)
    print(" RQ1 实验完成!")
    print(f"   - 结果保存至: {validator.output_dir}")
    print(f"\n 提示: 运行 'python step1_visualize.py' 生成可视化图表")
    print("="*60)
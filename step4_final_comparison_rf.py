"""
对比weighted特征集与LightGBM性能
修复：保持原始名称，但通过 (subset_name, method_type) 复合键区分数据
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve, auc,classification_report
)
from scipy.stats import ttest_rel, wilcoxon

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from utils import get_data


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.feature_names]


class WeightedMethodsEvaluator:
    def __init__(self, 
                 best_path,
                 step1_cv_path,
                 step1_test_path,
                 lightgbm_name="LightGBM",
                 output_dir="results"):

        self.best_path = Path(best_path)
        self.step1_cv_path = Path(step1_cv_path)
        self.step1_test_path = Path(step1_test_path)
        self.lightgbm_name = lightgbm_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        print("="*60)
        print("Step1: 加载数据（复用step1逻辑）")
        print("="*60)
        self.X_train, self.y_train, self.X_test, self.y_test = get_data()
        print(f"✓ 训练数据: X={self.X_train.shape}, y={self.y_train.shape}")
        print(f"✓ 测试数据: X={self.X_test.shape}, y={self.y_test.shape}")

        print("\n" + "="*60)
        print("Step2: 加载所有加权特征集")
        print("="*60)
        self._load_weighted_features()

        # RandomForest 参数
        self.rf_base_params = {
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1,
        }
        
        # GridSearch 参数网格
        self.param_grid = {
            'model__n_estimators': [500],
            'model__max_features': ['sqrt'],
            'model__max_depth': [15]
    }

        self._load_step1_results()
        
        # 使用复合键：(subset_name, method_type) 来存储结果
        self.cv_results_all = {}
        self.test_results_all = {}

    def _load_weighted_features(self):
        """加载所有特征集（保持原始名称）"""
        # 
        with open(self.best_path, "rb") as f:
            best_data = pickle.load(f)
        self.best_feats = best_data['features']
        self.best_name = best_data['subset_name']
        print(f"✓ 加载{self.best_name}: {len(self.best_feats)}个特征（最优）")


    def _load_step1_results(self):
        """加载step1的LightGBM交叉验证和测试结果"""
        with open(self.step1_cv_path, "rb") as f:
            step1_data = pickle.load(f)
        
        if self.lightgbm_name in step1_data['cv_results']:
            self.lightgbm_cv_results = step1_data['cv_results'][self.lightgbm_name]
            print(f"\n✓ 已加载 {self.lightgbm_name} 的交叉验证结果（{len(self.lightgbm_cv_results)} folds）")
        else:
            raise ValueError(f"未找到 {self.lightgbm_name} 的交叉验证结果")

        self.step1_test_df = pd.read_csv(self.step1_test_path)
        self.lightgbm_test_result = self.step1_test_df[
            self.step1_test_df['Subset'] == self.lightgbm_name
        ]
        if self.lightgbm_test_result.empty:
            raise ValueError(f"未找到 {self.lightgbm_name} 的测试集结果")
        print(f"✓ 已加载 {self.lightgbm_name} 的测试集结果")

    def cross_validate_single_subset(self, subset_name, features, method_type, n_folds=5):
        """对单个特征集进行五折交叉验证"""
        print(f"\n{'='*60}")
        print(f"交叉验证: {subset_name} ({len(features)} 个特征) - {method_type}")
        print(f"{'='*60}")
        
        cv_results = []
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(self.X_train, self.y_train)):
            X_tr, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_tr, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            
            pipeline = Pipeline([
                ('feature_selector', FeatureSelector(features)),
                ('model', RandomForestClassifier(**self.rf_base_params))
            ])
            
            gs = GridSearchCV(
                pipeline,
                param_grid=self.param_grid,
                scoring='roc_auc',
                cv=3,
                n_jobs=-1,
                verbose=0,
                return_train_score=True
            )
            
            gs.fit(X_tr, y_tr)
            
            best_model = gs.best_estimator_
            y_val_pred_proba = best_model.predict_proba(X_val)[:, 1]
            y_val_pred = (y_val_pred_proba > 0.5).astype(int)
            
            cm_fold = confusion_matrix(y_val, y_val_pred)
            tn_fold, fp_fold, fn_fold, tp_fold = cm_fold.ravel()
            tpr_fold = tp_fold / (tp_fold + fn_fold) if (tp_fold + fn_fold) > 0 else 0.0
            fpr_fold = fp_fold / (fp_fold + tn_fold) if (fp_fold + tn_fold) > 0 else 0.0
            
            fold_val_auc = roc_auc_score(y_val, y_val_pred_proba)
            fold_train_auc = roc_auc_score(y_tr, best_model.predict_proba(X_tr)[:, 1])
            rmse_fold = np.sqrt(np.mean((y_val - y_val_pred_proba) ** 2))

            fpr_curve, tpr_curve, _ = roc_curve(y_val, y_val_pred_proba)
            roc_auc_curve = auc(fpr_curve, tpr_curve)

            cv_results.append({
                'Subset': subset_name,
                'Method_Type': method_type,
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

            print(f"  Fold {fold_idx+1}: Val AUC={fold_val_auc:.4f}, "
                  f"Recall={tpr_fold:.4f}, FPR={fpr_fold:.4f}, RMSE={rmse_fold:.4f}")
        
        return cv_results

    def evaluate_on_test(self, subset_name, features, method_type):
        """在测试集上评估"""
        print(f"\n测试集评估: {subset_name} - {method_type}")

        pipeline = Pipeline([
            ('feature_selector', FeatureSelector(features)),
            ('model', RandomForestClassifier(**self.rf_base_params))
        ])
        
        gs = GridSearchCV(
            pipeline,
            param_grid=self.param_grid,
            scoring='roc_auc',
            cv=3,
            n_jobs=-1,
            verbose=0
        )
        
        gs.fit(self.X_train, self.y_train)
        
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

        class_report = classification_report(
            self.y_test, y_pred,
            target_names=['Class 0', 'Class 1'],  # 可根据实际标签修改（如'负例'、'正例'）
            output_dict=False,  # 设为True则返回字典，False返回字符串（更易读）
            digits=4  # 保留4位小数，精度更高
        )

        test_results = {
            'Subset': subset_name,
            'Method_Type': method_type,
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
            'roc_auc_test': roc_auc_test,
            'Classification_Report': class_report
        }

        print(f"  测试集 ROC-AUC: {test_results['Test_ROC_AUC']:.4f}")
        print(f"  Best params: {test_results['Best_Params']}")
        print(f"\n  测试集分类报告:")
        print("  " + "-"*80)  # 分隔线，美化输出
        # 按行打印分类报告，每行前加2个空格对齐
        for line in class_report.split('\n'):
            if line.strip():  # 跳过空行
                print(f"  {line}")
        print("  " + "-"*80)

        return test_results
    
    def run_all_weighted_methods(self):
        print("\n" + "="*60)
        print("Step3: 评估加权特征集（交叉验证 + 测试集）")
        print("="*60)

        feature_configs = [(self.best_name, self.best_feats, 'original')]

        for subset_name, features, method_type in feature_configs:
            key = (subset_name, method_type)
            cv_result = self.cross_validate_single_subset(subset_name, features, method_type)
            self.cv_results_all[key] = cv_result

            test_result = self.evaluate_on_test(subset_name, features, method_type)
            self.test_results_all[key] = test_result

        self._save_cv_and_test_results()


    def _save_cv_and_test_results(self):
        """保存交叉验证和测试集结果"""
        print("\n" + "="*60)
        print("保存交叉验证和测试集结果...")
        print("="*60)

        # === 1. 交叉验证汇总 ===
        cv_summary = []
        for (subset, method_type), results in self.cv_results_all.items():
            fold_metrics_df = pd.DataFrame(results)
            mean_metrics = fold_metrics_df.mean(numeric_only=True)
            std_metrics = fold_metrics_df.std(numeric_only=True)
            
            cv_summary.append({
                'Subset': subset,
                'Method_Type': method_type,
                'Mean_Train_ROC_AUC': mean_metrics['Fold_Train_ROC_AUC'],
                'Mean_Val_ROC_AUC': mean_metrics['Fold_Val_ROC_AUC'],
                'Std_Val_ROC_AUC': std_metrics['Fold_Val_ROC_AUC'],
                'Mean_Val_Recall': mean_metrics['Fold_Val_Recall'],
                'Mean_Val_FPR': mean_metrics['Fold_Val_FPR'],
                'Mean_Val_RMSE': mean_metrics['Fold_Val_RMSE'],
                'Features': fold_metrics_df['Features'].iloc[0]
            })
        
        cv_summary_df = pd.DataFrame(cv_summary)
        cv_summary_df.to_csv(self.output_dir / 'weighted_methods_cv_metrics_rf.csv', index=False)
        print(f"✓ 交叉验证汇总: {self.output_dir / 'weighted_methods_cv_metrics_rf.csv'}")

        # === 2. 测试集结果 ===
        test_df = pd.DataFrame(self.test_results_all.values())
        test_df.to_csv(self.output_dir / 'weighted_methods_test_metrics_rf.csv', index=False)
        print(f"✓ 测试集结果: {self.output_dir / 'weighted_methods_test_metrics_rf.csv'}")

        # === 3. 详细结果（PKL）===
        with open(self.output_dir / 'weighted_methods_cv_detailed_rf.pkl', 'wb') as f:
            pickle.dump({
                'cv_results': self.cv_results_all,
                'test_results': self.test_results_all,
            }, f)
        print(f"✓ 详细结果: {self.output_dir / 'weighted_methods_cv_detailed_rf.pkl'}")

    def prepare_comparison_data(self):
        """准备与LightGBM对比的数据"""
        print("\n" + "="*60)
        print("Step4: 准备对比数据")
        print("="*60)

        lightgbm_cv_df = pd.DataFrame(self.lightgbm_cv_results)
        
        all_cv_data = []
        for (subset_name, method_type), cv_results in self.cv_results_all.items():
            cv_df = pd.DataFrame(cv_results)
            all_cv_data.append(cv_df)
        
        weighted_cv_df = pd.concat(all_cv_data, ignore_index=True)
        comparison_cv_df = pd.concat([lightgbm_cv_df, weighted_cv_df], ignore_index=True)
        
        comparison_cv_df.to_csv(
            self.output_dir / 'all_methods_cv_folds_rf.csv', 
            index=False
        )
        print(f"✓ Fold级别对比数据: {self.output_dir / 'all_methods_cv_folds_rf.csv'}")

        return comparison_cv_df

    def compare_with_lightgbm(self):
        """与LightGBM进行详细对比"""
        print("\n" + "="*80)
        print("Step5: 与LightGBM对比分析")
        print("="*80)

        lgbm_test = self.lightgbm_test_result.iloc[0].to_dict()
        lgbm_clean = {
            'Subset': lgbm_test['Subset'],
            'Method_Type': None,
            'Features': lgbm_test['Features'],
            'Train_ROC_AUC': lgbm_test['Train_ROC_AUC'],
            'Test_ROC_AUC': lgbm_test['Test_ROC_AUC'],
            'Recall': lgbm_test['Recall'],
            'FPR': lgbm_test['FPR'],
            'RMSE': lgbm_test['RMSE'],
            'Best_Params': lgbm_test.get('Best_Params', ''),
            'Classification_Report': lgbm_test.get('Classification_Report', 'Unknown') 
        }

        weighted_test = list(self.test_results_all.values())[0]

        df_compare = pd.DataFrame([lgbm_clean, weighted_test])
        df_compare = df_compare.sort_values('Test_ROC_AUC', ascending=False).reset_index(drop=True)
        df_compare.insert(0, 'Rank', range(1, len(df_compare)+1))

        print("\n性能对比表（LightGBM vs WeightedInt）:")
        print(df_compare[['Rank','Subset','Features','Train_ROC_AUC','Test_ROC_AUC','Recall','FPR','RMSE']].to_string(index=False))

        output_path = self.output_dir / 'WeightedInt_vs_LightGBM_rf.csv'
        df_compare.to_csv(output_path, index=False)
        print(f"✓ 对比结果已保存至: {output_path}")

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

class RQ2Visualizer:
    def __init__(self, results_dir, fig_dir):
        self.results_dir = Path(results_dir)
        self.fig_dir = Path(fig_dir)
        self.fig_dir.mkdir(exist_ok=True)
        self._load_results()

    def _load_results(self):
        print("加载结果文件中...")
        weighted_df = pd.read_csv(self.results_dir / "weighted_methods_test_metrics_rf.csv")
        rq1_df = pd.read_csv(self.results_dir / "rq1_test_metrics_rf.csv")

        weighted = weighted_df[weighted_df['Subset'] == 'WeightedInt_39'].iloc[0]
        lightgbm = rq1_df[rq1_df['Subset'] == 'LightGBM'].iloc[0]

        self.data = {
            'WeightedInt_39': {
                'y_test': eval(weighted['y_test']),
                'y_pred_proba': eval(weighted['y_test_pred_proba']),
                'auc': weighted['Test_ROC_AUC']
            },
            'LightGBM': {
                'y_test': eval(lightgbm['y_test']),
                'y_pred_proba': eval(lightgbm['y_test_pred_proba']),
                'auc': lightgbm['Test_ROC_AUC']
            }
        }
        print("✓ 加载成功")

    def plot_comparison(self):
        plt.figure(figsize=(8,6))
        plt.plot([0,1],[0,1],linestyle='--',color='gray',label='Chance', lw=1.5)

        linestyles = {
        'WeightedInt_39': '-',    # 实线
        'LightGBM': '--'          # 虚线
        # 可选样式：'-.'（点划线）、':'（点线）等
    }

        for name, result in self.data.items():
            fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
            plt.plot(fpr, tpr, linestyle=linestyles[name], lw=2, label=f"{name} (AUC={result['auc']:.4f})")

        plt.title("ROC Curve Comparison: Weighted Integrated vs. LightGBM-based Subsets", fontsize=12)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.grid(False)

        save_path = self.fig_dir / "ROC_WeightedInt39_vs_LightGBM.png"
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        print(f"✓ 已保存图像至: {save_path}")
        plt.show()


# ===============================
# 主执行入口
# ===============================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("WeightedInt_39 vs LightGBM 评估实验启动")
    print("="*60)

    evaluator = WeightedMethodsEvaluator(
        best_path="/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/Project/data/original_features_39.pkl",
        step1_cv_path="/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/Project/results/cv_results_detailed_rf.pkl",
        step1_test_path="/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/Project/results/rq1_test_metrics_rf.csv",
        lightgbm_name="LightGBM",
        output_dir="/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/Project/results"
    )
    test_df = pd.read_csv(evaluator.output_dir / 'weighted_methods_test_metrics_rf.csv')
    for _, row in test_df.iterrows():
        key = (row['Subset'], row['Method_Type'])
        evaluator.test_results_all[key] = row.to_dict()
    evaluator.compare_with_lightgbm()

    visualizer = RQ2Visualizer(
        results_dir="/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/Project/results",
        fig_dir="/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/Project/figures"
    )
    visualizer.plot_comparison()

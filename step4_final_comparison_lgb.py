
"""
对比方法二（after_removal）和方法三（after_replacement）的特征集与LightGBM性能
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
    confusion_matrix, roc_auc_score, roc_curve, auc
)
from scipy.stats import ttest_rel, wilcoxon

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
                 ar_best_path,
                 ar_42_path,
                 arpl_best_path,
                 arpl_42_path,
                 step1_cv_path,
                 step1_test_path,
                 lightgbm_name="LightGBM",
                 output_dir="results"):

        self.ar_best_path = Path(ar_best_path)
        self.ar_42_path = Path(ar_42_path)
        self.arpl_best_path = Path(arpl_best_path)
        self.arpl_42_path = Path(arpl_42_path)
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

        self.lgb_base_params = {
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        self.param_grid = {
            'model__n_estimators': [400, 500],
            'model__max_depth': [6],
            'model__learning_rate': [0.1],
        }

        self._load_step1_results()
        
        # 使用复合键：(subset_name, method_type) 来存储结果
        self.cv_results_all = {}
        self.test_results_all = {}

    def _load_weighted_features(self):
        """加载方法二和方法三的所有特征集（保持原始名称）"""
        # 方法二（after_removal）
        with open(self.ar_best_path, "rb") as f:
            ar_best_data = pickle.load(f)
        self.ar_best_feats = ar_best_data['features']
        self.ar_best_name = ar_best_data['subset_name']
        print(f"✓ 加载{self.ar_best_name}: {len(self.ar_best_feats)}个特征（after_removal最优）")

        with open(self.ar_42_path, "rb") as f:
            ar_42_data = pickle.load(f)
        self.ar_42_feats = ar_42_data['features']
        self.ar_42_name = ar_42_data['subset_name']
        print(f"✓ 加载{self.ar_42_name}: {len(self.ar_42_feats)}个特征（after_removal固定42）")

        # 方法三（after_replacement）
        with open(self.arpl_best_path, "rb") as f:
            arpl_best_data = pickle.load(f)
        self.arpl_best_feats = arpl_best_data['features']
        self.arpl_best_name = arpl_best_data['subset_name']
        print(f"✓ 加载{self.arpl_best_name}: {len(self.arpl_best_feats)}个特征（after_replacement最优）")

        with open(self.arpl_42_path, "rb") as f:
            arpl_42_data = pickle.load(f)
        self.arpl_42_feats = arpl_42_data['features']
        self.arpl_42_name = arpl_42_data['subset_name']
        print(f"✓ 加载{self.arpl_42_name}: {len(self.arpl_42_feats)}个特征（after_replacement固定42）")

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
                ('model', LGBMClassifier(**self.lgb_base_params))
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
            ('model', LGBMClassifier(**self.lgb_base_params))
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
            'roc_auc_test': roc_auc_test
        }

        print(f"  测试集 ROC-AUC: {test_results['Test_ROC_AUC']:.4f}")
        print(f"  Best params: {test_results['Best_Params']}")

        return test_results

    def run_all_weighted_methods(self):
        """对所有加权特征集进行交叉验证和测试集评估"""
        print("\n" + "="*60)
        print("Step3: 评估所有加权特征集（交叉验证 + 测试集）")
        print("="*60)

        feature_configs = [
            (self.ar_best_name, self.ar_best_feats, "after_removal"),
            (self.ar_42_name, self.ar_42_feats, "after_removal"),
            (self.arpl_best_name, self.arpl_best_feats, "after_replacement"),
            (self.arpl_42_name, self.arpl_42_feats, "after_replacement")
        ]

        for subset_name, features, method_type in feature_configs:
            # 使用复合键存储：(subset_name, method_type)
            key = (subset_name, method_type)
            
            cv_result = self.cross_validate_single_subset(
                subset_name, features, method_type
            )
            self.cv_results_all[key] = cv_result
            
            test_result = self.evaluate_on_test(
                subset_name, features, method_type
            )
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
        cv_summary_df.to_csv(self.output_dir / 'weighted_methods_cv_metrics.csv', index=False)
        print(f"✓ 交叉验证汇总: {self.output_dir / 'weighted_methods_cv_metrics.csv'}")

        # === 2. 测试集结果 ===
        test_df = pd.DataFrame(self.test_results_all.values())
        test_df.to_csv(self.output_dir / 'weighted_methods_test_metrics.csv', index=False)
        print(f"✓ 测试集结果: {self.output_dir / 'weighted_methods_test_metrics.csv'}")

        # === 3. 详细结果（PKL）===
        with open(self.output_dir / 'weighted_methods_cv_detailed.pkl', 'wb') as f:
            pickle.dump({
                'cv_results': self.cv_results_all,
                'test_results': self.test_results_all,
            }, f)
        print(f"✓ 详细结果: {self.output_dir / 'weighted_methods_cv_detailed.pkl'}")

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
            self.output_dir / 'all_methods_cv_folds.csv', 
            index=False
        )
        print(f"✓ Fold级别对比数据: {self.output_dir / 'all_methods_cv_folds.csv'}")

        return comparison_cv_df

    def compare_with_lightgbm(self):
        """与LightGBM进行详细对比（分方法二和方法三）"""
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
            'Best_Params': lgbm_test.get('Best_Params', '')
        }

        # 方法二（after_removal）对比
        print("\n" + "="*60)
        print("方法二（after_removal）与LightGBM对比")
        print("="*60)
        
        ar_methods_data = [self.test_results_all[k] for k in self.test_results_all.keys() 
                           if k[1] == 'after_removal']
        ar_compare_data = [lgbm_clean] + ar_methods_data
        ar_compare_df = pd.DataFrame(ar_compare_data)
        ar_compare_df = ar_compare_df.sort_values('Test_ROC_AUC', ascending=False).reset_index(drop=True)
        ar_compare_df.insert(0, 'Rank', range(1, len(ar_compare_df)+1))

        print(f"\n性能对比表（包含 {len(ar_methods_data)} 个加权方法 + LightGBM）:")
        print("-"*150)
        pd.set_option('display.float_format', '{:.4f}'.format)
        print(ar_compare_df[['Rank', 'Subset', 'Method_Type', 'Features', 'Train_ROC_AUC', 
                             'Test_ROC_AUC', 'Recall', 'FPR', 'RMSE']].to_string(index=False))
        print("-"*150)

        ar_output = self.output_dir / 'after_removal_vs_lightgbm.csv'
        ar_compare_df.to_csv(ar_output, index=False)

        # 方法三（after_replacement）对比
        print("\n" + "="*60)
        print("方法三（after_replacement）与LightGBM对比")
        print("="*60)
        
        arpl_methods_data = [self.test_results_all[k] for k in self.test_results_all.keys() 
                             if k[1] == 'after_replacement']
        arpl_compare_data = [lgbm_clean] + arpl_methods_data
        arpl_compare_df = pd.DataFrame(arpl_compare_data)
        arpl_compare_df = arpl_compare_df.sort_values('Test_ROC_AUC', ascending=False).reset_index(drop=True)
        arpl_compare_df.insert(0, 'Rank', range(1, len(arpl_compare_df)+1))

        print(f"\n性能对比表（包含 {len(arpl_methods_data)} 个加权方法 + LightGBM）:")
        print("-"*150)
        print(arpl_compare_df[['Rank', 'Subset', 'Method_Type', 'Features', 'Train_ROC_AUC', 
                              'Test_ROC_AUC', 'Recall', 'FPR', 'RMSE']].to_string(index=False))
        print("-"*150)

        arpl_output = self.output_dir / 'after_replacement_vs_lightgbm.csv'
        arpl_compare_df.to_csv(arpl_output, index=False)

        print(f"\n✓ 对比结果已保存至:")
        print(f"   {ar_output}")
        print(f"   {arpl_output}")

    def perform_significance_tests(self):
        """执行显著性检验：只对比 WeightedInt_50 和 WeightedInt_47 vs LightGBM"""
        print("\n" + "="*80)
        print("Step6: 显著性检验（Paired t-test & Wilcoxon signed-rank test）")
        print("仅对比最优特征集：WeightedInt_50（方法三）和 WeightedInt_47（方法二）")
        print("="*80)

        cv_folds_df = pd.read_csv(self.output_dir / 'all_methods_cv_folds.csv')
        
        metrics = ['Fold_Val_ROC_AUC', 'Fold_Val_Recall']
        baseline_model = self.lightgbm_name
        
        # 只测试这两个最优特征集
        target_subsets = [
            ('WeightedInt_47', 'after_removal', '方法二最优'),
            ('WeightedInt_50', 'after_replacement', '方法三最优')
        ]
        
        all_results = []

        for subset_name, method_type, description in target_subsets:
            # 检查该特征集是否存在
            if (subset_name, method_type) not in self.cv_results_all:
                print(f"\n⚠️  警告: {subset_name} ({method_type}) 未找到，跳过")
                continue
                
            print(f"\n{'='*70}")
            print(f"对比: {subset_name} ({description}) vs {baseline_model}")
            print(f"{'='*70}")
            
            for metric in metrics:
                # 获取加权方法的fold数据
                df_w = cv_folds_df[
                    (cv_folds_df['Subset'] == subset_name) & 
                    (cv_folds_df['Method_Type'] == method_type)
                ].sort_values('Fold_Index')[metric].values
                
                # 获取baseline的fold数据
                df_b = cv_folds_df[
                    cv_folds_df['Subset'] == baseline_model
                ].sort_values('Fold_Index')[metric].values
                
                min_len = min(len(df_w), len(df_b))
                if min_len == 0:
                    print(f"  ⚠️  {metric}: 数据不足，跳过")
                    continue
                    
                df_w, df_b = df_w[:min_len], df_b[:min_len]
                
                # Paired t-test（单侧检验：加权方法 > baseline）
                t_stat, p_t = ttest_rel(df_w, df_b, alternative='greater')
                
                # Wilcoxon signed-rank test（非参数检验）
                try:
                    w_stat, p_w = wilcoxon(df_w, df_b, alternative='greater')
                except ValueError:
                    w_stat, p_w = None, None
                
                mean_w, mean_b = df_w.mean(), df_b.mean()
                gain = (mean_w - mean_b) / mean_b * 100 if mean_b != 0 else 0
                
                # 显著性判断（α = 0.05）
                is_significant = p_t < 0.05
                significance = "✓ 显著提升" if is_significant else "✗ 无显著差异"
                
                all_results.append({
                    "Subset": subset_name,
                    "Description": description,
                    "Method_Type": method_type,
                    "Metric": metric,
                    "Mean_Weighted": round(mean_w, 6),
                    "Mean_Baseline": round(mean_b, 6),
                    "Relative_Improvement(%)": round(gain, 3),
                    "Paired_t_pvalue": round(p_t, 6),
                    "Wilcoxon_pvalue": round(p_w, 6) if p_w is not None else None,
                    "Significance(α=0.05)": significance
                })
                
                print(f"\n  📊 {metric}:")
                print(f"    Mean {subset_name}: {mean_w:.6f}")
                print(f"    Mean {baseline_model}: {mean_b:.6f}")
                print(f"    Absolute Difference: {mean_w - mean_b:+.6f}")
                print(f"    Relative Improvement: {gain:+.3f}%")
                print(f"    Paired t-test p-value: {p_t:.6f}")
                
                wilcoxon_display = f"{p_w:.6f}" if p_w is not None else "N/A"
                print(f"    Wilcoxon p-value: {wilcoxon_display}")
                print(f"    结论: {significance}")

        # 保存结果
        summary = pd.DataFrame(all_results)
        save_path = self.output_dir / "significance_test_summary.csv"
        summary.to_csv(save_path, index=False)
        
        print("\n" + "="*80)
        print("📋 显著性检验结果汇总：")
        print("="*80)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(summary.to_string(index=False))
        print(f"\n✅ 显著性检验结果已保存至: {save_path}")
        
        # 打印最终结论
        print("\n" + "="*80)
        print("🎯 最终结论：")
        print("="*80)
        
        for subset_name, method_type, description in target_subsets:
            if (subset_name, method_type) not in self.cv_results_all:
                continue
                
            subset_results = summary[summary['Subset'] == subset_name]
            if subset_results.empty:
                continue
                
            significant_metrics = subset_results[
                subset_results['Significance(α=0.05)'] == '✓ 显著提升'
            ]['Metric'].tolist()
            
            print(f"\n  {subset_name} ({description}):")
            if significant_metrics:
                print(f"    ✅ 在以下指标上显著优于 {baseline_model}:")
                for m in significant_metrics:
                    row = subset_results[subset_results['Metric'] == m].iloc[0]
                    print(f"       - {m}: 提升 {row['Relative_Improvement(%)']:.3f}% (p={row['Paired_t_pvalue']:.4f})")
            else:
                print(f"    ❌ 与 {baseline_model} 无显著差异")
        
        print("="*80)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("方法二/三特征集完整评估实验")
    print("包含：交叉验证 + 测试集评估 + 显著性检验")
    print("="*60)
    
    evaluator = WeightedMethodsEvaluator(
        ar_best_path='/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/Project/data/after_removal_features_47.pkl',
        ar_42_path='/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/Project/data/after_removal_features_42.pkl',
        arpl_best_path='/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/Project/data/after_replacement_features_50.pkl',
        arpl_42_path='/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/Project/data/after_replacement_features_42.pkl',
        step1_cv_path='/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/Project/results/cv_results_detailed.pkl',
        step1_test_path='/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/Project/results/rq1_test_metrics.csv',
        lightgbm_name="LightGBM",
        output_dir='/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/Project/results'
    )
    
    evaluator.run_all_weighted_methods()
    evaluator.prepare_comparison_data()
    evaluator.compare_with_lightgbm()
    evaluator.perform_significance_tests()
    
    print("\n" + "="*60)
    print("✅ 完整评估实验完成!")
    print(f"   结果保存至: {evaluator.output_dir}")
    print("\n生成的文件:")
    print("   - weighted_methods_cv_metrics.csv（交叉验证汇总）")
    print("   - weighted_methods_test_metrics.csv（测试集结果）")
    print("   - weighted_methods_cv_detailed.pkl（详细结果）")
    print("   - all_methods_cv_folds.csv（Fold级别对比数据）")
    print("   - after_removal_vs_lightgbm.csv（方法二对比）")
    print("   - after_replacement_vs_lightgbm.csv（方法三对比）")
    print("   - significance_test_summary.csv（显著性检验）")
    print("="*60)
import numpy as np
import pandas as pd
import pickle
from scipy.optimize import minimize
from scipy.stats import pearsonr
from pathlib import Path
import matplotlib.pyplot as plt


class WeightOptimizer:
    def __init__(self,
                 cv_ranked_path='results/rq1_cv_metrics_ranked_rf.csv',
                 output_path='results/optimized_weights_rf.pkl'):
        self.cv_ranked_path = Path(cv_ranked_path)
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(exist_ok=True)

        # 定义指标
        self.high_value_metrics = ["Mean_Val_ROC_AUC", "Mean_Val_Recall"]
        self.low_value_metrics = ["Mean_Val_FPR", "Features"]
        self.all_metrics = self.high_value_metrics + self.low_value_metrics

        # 数据容器
        self.results = {
            'original': {  # 原始处理（不处理全零子集）
                'methods': None,
                'data': None,
                'standardized_data': None,
                'metric_weights': None,
                'subset_weights': None,
                'weighted_feature_count': None,
                'gamma': None
            },
            'after_removal': {  # 删除全零子集后重新计算
                'methods': None,
                'data': None,
                'standardized_data': None,
                'metric_weights': None,
                'subset_weights': None,
                'weighted_feature_count': None,
                'gamma': None
            },
            'after_replacement': {  # 替换全零子集为下一名
                'methods': None,
                'data': None,
                'standardized_data': None,
                'metric_weights': None,
                'subset_weights': None,
                'weighted_feature_count': None,
                'gamma': None
            }
        }

        self.ranked_df = None
        self.best_gamma = {
            'original': None,
            'after_removal': None,
            'after_replacement': None
        }
        self.zero_subsets = None  # 存储原始处理中检测到的全零子集

    # 数据加载基础方法
    def _load_ranked_data(self):
        if self.ranked_df is None:
            print(f"从 {self.cv_ranked_path} 加载排序后的交叉验证结果...")
            self.ranked_df = pd.read_csv(self.cv_ranked_path)
            numeric_cols = self.ranked_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            self.ranked_df[numeric_cols] = self.ranked_df[numeric_cols].round(4)
            required_cols = ['Subset'] + self.all_metrics
            missing_cols = [c for c in required_cols if c not in self.ranked_df.columns]
            if missing_cols:
                raise ValueError(f"缺少列: {missing_cols}")
        return self.ranked_df

    # 加载指定的子集列表
    def _load_specified_subsets(self, key, methods):
        """加载指定的子集列表，不做全零检测"""
        ranked_df = self._load_ranked_data()
        self.results[key]['methods'] = methods
        self.results[key]['data'] = ranked_df[ranked_df['Subset'].isin(methods)].set_index('Subset')
        # 确保子集顺序与输入methods一致
        self.results[key]['data'] = self.results[key]['data'].reindex(methods)
        print(f"{key} 加载子集: {methods}")
        return self

    # 标准化数据
    def _standardize_data(self, data):
        standardized = pd.DataFrame(index=data.index)
        # 高值优指标
        for col in self.high_value_metrics:
            x_min, x_max = data[col].min(), data[col].max()
            if x_max != x_min:
                standardized[col] = (data[col] - x_min) / (x_max - x_min)
            else:
                standardized[col] = 1.0
        # 低值优指标
        for col in self.low_value_metrics:
            x_min, x_max = data[col].min(), data[col].max()
            if x_max != x_min:
                standardized[col] = (x_max - data[col]) / (x_max - x_min)
            else:
                standardized[col] = 1.0
        return standardized.round(4)

    # 准备优化数据
    def prepare_optimization_data(self, key):
        if self.results[key]['data'] is None:
            raise ValueError(f"请先加载 {key} 的子集数据")
        self.results[key]['standardized_data'] = self._standardize_data(self.results[key]['data'])
        print(f"\n{key} 标准化结果：")
        print(self.results[key]['standardized_data'].to_string())
        return self

    # 目标函数
    def _objective_function(self, weights, key):
        X = self.results[key]['standardized_data'].values
        metrics = self.all_metrics
        S = np.dot(X, weights)

        auc_col = X[:, metrics.index('Mean_Val_ROC_AUC')]
        rec_col = X[:, metrics.index('Mean_Val_Recall')]
        fpr_col = X[:, metrics.index('Mean_Val_FPR')]
        feat_col = X[:, metrics.index('Features')]

        def safe_corr(a, b):
            try:
                r, _ = pearsonr(a, b)
                return r if np.isfinite(r) else 0
            except Exception:
                return 0

        corr_auc = safe_corr(S, auc_col)
        corr_rec = safe_corr(S, rec_col)
        corr_fpr = safe_corr(S, fpr_col)
        corr_feat = safe_corr(S, feat_col)

        lam1, lam2, lam3, lam4, lam5 = 1.0, 1.2, 0.5, 0.1, 1e-3
        J = (lam1 * corr_auc + lam2 * corr_rec
             - lam3 * (corr_fpr + corr_feat)
             + lam4 * np.std(S)
             - lam5 * np.sum(weights **2))
        return -J

    # 优化权重
    def optimize_weights(self, key):
        if self.results[key]['standardized_data'] is None:
            self.prepare_optimization_data(key)

        gammas = np.linspace(start=0.55, stop=0.75, num=5)  # 生成连续区间：0.5,0.55,0.6,...,0.75

        best_result, best_fun = None, -np.inf

        for gamma in gammas:
            print(f"\n{key} - 尝试 γ = {gamma:.2f} 权重约束...")  # 保留2位小数，适配步长
            n_metrics = len(self.all_metrics)
            initial_weights = np.ones(n_metrics) / n_metrics

        # for gamma in gammas:
        #     print(f"\n{key} - 尝试 γ = {gamma:.1f} 权重约束...")
        #     n_metrics = len(self.all_metrics)
        #     initial_weights = np.ones(n_metrics) / n_metrics

            constraints = [
                {'type': 'eq', 'fun': lambda w: w[0] + w[1] - gamma},
                {'type': 'eq', 'fun': lambda w: w[2] + w[3] - (1 - gamma)},
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
            ]
            bounds = [(0.1, 1.0)] * n_metrics

            result = minimize(
                lambda w: self._objective_function(w, key),
                initial_weights,
                method='SLSQP',
                constraints=constraints,
                bounds=bounds,
                options={'maxiter': 1000, 'disp': False}
            )

            if result.success:
                val = -result.fun
                print(f"  {key} - γ={gamma:.2f} -> 目标函数值: {val:.6f}")
                if val > best_fun:
                    best_fun = val
                    best_result = result
                    self.best_gamma[key] = gamma

        if best_result is None:
            raise RuntimeError(f"{key} 所有γ优化均失败")

        print(f"\n{key} 最优 γ = {self.best_gamma[key]:.2f}，目标函数值 = {best_fun:.6f}")
        self.results[key]['metric_weights'] = dict(zip(self.all_metrics, best_result.x.round(4)))
        print(f"{key} 最终最优指标权重：")
        for k, v in self.results[key]['metric_weights'].items():
            print(f"  {k}: {v:.4f}")
        return self

    
    def calculate_subset_weights(self, key):
        weights_array = np.array(list(self.results[key]['metric_weights'].values()))
        standardized_data = self.results[key]['standardized_data']
        subset_scores = np.dot(standardized_data.values, weights_array)
        
        # Calculate total score
        total_score = np.sum(subset_scores)
        print(f"\nTotal score of all subsets: {total_score:.4f}")
        
        # Calculate initial weights (score ratio, rounded to 2 decimals)
        subset_weights = (subset_scores / total_score).round(2)
        subset_dict = dict(zip(standardized_data.index, subset_weights))

        # Calculate weight sum bias and adjust
        current_sum = sum(subset_dict.values())
        bias = 1.0 - current_sum  # Bias = target sum (1.0) - current sum
        
        # Adjust only if bias is significant
        if abs(bias) > 1e-6:
            # Find subset with the smallest weight (MI in this case)
            min_subset = min(subset_dict, key=subset_dict.get)
            # Add bias to the smallest weight (MI: 0.03 + 0.01 = 0.04)
            subset_dict[min_subset] = round(subset_dict[min_subset] + bias, 2)
            
            # Ensure no negative weight (redundant here but safe)
            if subset_dict[min_subset] < 0:
                subset_dict[min_subset] = 0.01
                max_subset = max(subset_dict, key=subset_dict.get)
                subset_dict[max_subset] = round(subset_dict[max_subset] - abs(bias), 2)
        
        # Update results with adjusted weights
        self.results[key]['subset_weights'] = subset_dict
        
        # Print ADJUSTED weights (关键：调整后再打印)
        print("\nFinal weights of each subset (score ratio):")
        for subset, weight in subset_dict.items():
            print(f"  {subset}: {weight:.2f} ({weight:.2f}%)")  # 整数百分比显示
        
        # Verify adjusted total sum
        adjusted_sum = sum(subset_dict.values())
        print(f"\nWeight sum verification: {adjusted_sum:.2f} {'✓' if abs(adjusted_sum - 1.0) < 1e-6 else '✗'}")
        return self

    def calculate_weighted_features(self, key):
        total = 0.0
        # Use adjusted weights (MI=4%) for calculation
        for subset, weight in self.results[key]['subset_weights'].items():
            count = self.results[key]['data'].loc[subset, 'Features']
            total += round(count * weight)
        self.results[key]['weighted_feature_count'] = round(total)
        print(f"{key} Final weighted feature count: {self.results[key]['weighted_feature_count']}")
        return self

    # 保存所有结果
    def save_results(self):
        save_data = {k: v for k, v in self.results.items()}
        with open(self.output_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"\n所有结果已保存至: {self.output_path}")
        return self

    # 完整执行流程
    def run_optimization(self):
        print("=" * 80)
        print("开始执行改进版权重优化（按要求调整流程）")
        print("=" * 80)

        # 1. 原始处理：先处理前四名，不处理全零子集
        print("\n" + "=" * 60)
        print("处理方式1: 原始Top4（不处理全零子集）")
        print("=" * 60)
        ranked_df = self._load_ranked_data()
        original_methods = ranked_df['Subset'].head(4).tolist()  # 取原始前四名
        self._load_specified_subsets('original', original_methods) \
            .prepare_optimization_data('original')
        
        # 检测原始处理中的全零子集
        self.zero_subsets = [
            idx for idx in self.results['original']['standardized_data'].index 
            if (self.results['original']['standardized_data'].loc[idx] == 0).all()
        ]
        print(f"\n原始Top4中检测到的全零子集: {self.zero_subsets if self.zero_subsets else '无'}")
        
        # 完成原始处理的后续步骤
        self.optimize_weights('original') \
            .calculate_subset_weights('original') \
            .calculate_weighted_features('original')

        # 2. 方式1：删除全零子集后重新计算
        print("\n" + "=" * 60)
        print("处理方式2: 删除全零子集后重新标准化优化")
        print("=" * 60)
        if self.zero_subsets:
            # 从原始方法中删除全零子集
            removal_methods = [m for m in original_methods if m not in self.zero_subsets]
            if len(removal_methods) < 2:
                print("警告：删除全零子集后剩余不足2个，无法进行优化")
            else:
                self._load_specified_subsets('after_removal', removal_methods) \
                    .prepare_optimization_data('after_removal') \
                    .optimize_weights('after_removal') \
                    .calculate_subset_weights('after_removal') \
                    .calculate_weighted_features('after_removal')
        else:
            print("原始Top4中无全零子集，无需删除处理")

        # 3. 方式2：替换全零子集为下一名
        print("\n" + "=" * 60)
        print("处理方式3: 替换全零子集为下一名后重新计算")
        print("=" * 60)
        if self.zero_subsets:
            all_methods = ranked_df['Subset'].tolist()
            # 找到原始前四名之后的第一个替补索引
            last_original_idx = all_methods.index(original_methods[-1])
            replacement_methods = original_methods.copy()
            
            # 替换每个全零子集
            for zero_method in self.zero_subsets:
                # 寻找下一个替补方法（不在原始列表中）
                next_idx = last_original_idx + 1
                while next_idx < len(all_methods) and all_methods[next_idx] in replacement_methods:
                    next_idx += 1
                if next_idx >= len(all_methods):
                    print("警告：没有足够的替补子集，无法完成替换")
                    break
                
                # 执行替换
                replacement_methods.remove(zero_method)
                new_method = all_methods[next_idx]
                replacement_methods.append(new_method)
                last_original_idx = next_idx  # 更新最后索引
                print(f"替换全零子集 {zero_method} 为 {new_method}")
            
            # 执行替换后的优化
            self._load_specified_subsets('after_replacement', replacement_methods) \
                .prepare_optimization_data('after_replacement') \
                .optimize_weights('after_replacement') \
                .calculate_subset_weights('after_replacement') \
                .calculate_weighted_features('after_replacement')
        else:
            print("原始Top4中无全零子集，无需替换处理")

        # 保存所有结果
        self.save_results()
        print("\n" + "=" * 80)
        print("所有优化流程完成 ✅")
        print("=" * 80)


# 主程序入口
if __name__ == '__main__':
    optimizer = WeightOptimizer(
        cv_ranked_path='/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/Project/results/rq1_cv_metrics_ranked_rf.csv',
        output_path='/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/Project/results/rq2_optimized_weights_rf.pkl'
    )
    optimizer.run_optimization()

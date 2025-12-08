# step1_visualize.py
"""
RQ1 可视化脚本
- 读取 step1 保存的结果
- 生成各种可视化图表
- 可独立运行和修改
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('default')
sns.set_palette("husl")


class RQ1Visualizer:
    
    def __init__(self, results_dir='results', fig_dir='figures'):
        self.results_dir = Path(results_dir)
        self.fig_dir = Path(fig_dir)
        self.fig_dir.mkdir(exist_ok=True)
        
        # 加载数据
        self._load_results()
        # 初始化统一颜色字典
        self._init_color_dict()
        
    def _load_results(self):
        """加载实验结果"""
        print("="*60)
        print("加载实验结果...")
        print("="*60)
        
        # 加载交叉验证汇总
        self.cv_summary = pd.read_csv(self.results_dir / 'rq1_cv_metrics_rf.csv')
        print(f"加载交叉验证结果: {len(self.cv_summary)} 个方法")
        
        # 加载测试集结果
        self.test_results = pd.read_csv(self.results_dir / 'rq1_test_metrics_rf.csv')
        print(f"加载测试集结果: {len(self.test_results)} 个方法")
        
        # 加载详细结果
        with open(self.results_dir / 'cv_results_detailed_rf.pkl', 'rb') as f:
            detailed = pickle.load(f)
            self.cv_results = detailed['cv_results']  # 格式: {方法名: [每折结果列表]}
            self.test_results_detailed = detailed['test_results']  # 格式: {方法名: 测试集结果}
            
            # 从测试集结果中提取y_test（所有方法的y_test相同，取第一个即可）
            first_method = next(iter(self.test_results_detailed.keys()))
            self.y_test = self.test_results_detailed[first_method]['y_test']
        print(f"✓ 加载详细预测结果")
        
        print("\n可用的可视化方法:")
        print("  1. plot_cv_roc() - 交叉验证 ROC 曲线")
        print("  2. plot_test_roc() - 测试集 ROC 曲线")
        print("  3. plot_cv_combined() - 交叉验证综合对比图")
        print("  4. plot_test_comparison() - 测试集综合对比图")
        print("  5. plot_all() - 生成所有图表")
        print("="*60 + "\n")
    
    def _init_color_dict(self):
        """初始化所有子集的统一颜色字典"""
        all_subsets = list(self.cv_results.keys())
        cmap = plt.get_cmap("tab20")
        self.color_dict = {subset: cmap(i) for i, subset in enumerate(all_subsets)}
    
    def plot_cv_roc(self, save=True):
        # 绘制交叉验证 ROC 曲线
        print(" 生成交叉验证 ROC 曲线...")
        
        plt.figure(figsize=(10, 8))  
        
        # 使用统一颜色字典
        for subset in self.cv_results.keys():
            fold_results = self.cv_results[subset]

            # 每折ROC曲线插值
            tprs = []
            mean_fpr = np.linspace(0, 1, 100)
            for fold_result in fold_results:
                # 使用预存的FPR和TPR数据
                fpr = np.array(fold_result['Fold_FPR'])
                tpr = np.array(fold_result['Fold_TPR'])
                
                tpr_interp = np.interp(mean_fpr, fpr, tpr)
                tpr_interp[0] = 0.0
                tprs.append(tpr_interp)

            mean_tpr = np.mean(tprs, axis=0)
            # 使用预计算的AUC均值
            mean_auc = np.mean([fold['Fold_ROC_AUC'] for fold in fold_results])

            plt.plot(
                mean_fpr, mean_tpr, lw=2,
                label=f"{subset} (AUC = {mean_auc:.3f})",
                color=self.color_dict[subset]
            )

        plt.plot([0, 1], [0, 1], linestyle="--", color="grey", lw=2)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        # 标题
        plt.title("Cross-Validation Mean ROC Curves")
        plt.legend(loc="lower right")
        plt.grid(False)
        
        if save:
            save_path = self.fig_dir / 'rq1_cv_mean_roc_rf.png'
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f" 已保存: {save_path}")
            
            plt.show()
    
    def plot_cv_combined(self, save=True):
            """绘制组合图：特征数-AUC散点+多指标柱状图"""
            print("生成交叉验证组合图...")
            
            # 准备交叉验证数据
            cv_raw_data = []
            for subset, fold_results in self.cv_results.items():
                for fold in fold_results:
                    cv_raw_data.append({
                        "Subset_Name": subset,
                        "Feature_Count": fold["Features"],
                        "Fold_Val_AUC": fold["Fold_Val_ROC_AUC"],
                        "Fold_Val_Recall": fold["Fold_Val_Recall"],
                        "Fold_Val_RMSE": fold["Fold_Val_RMSE"]
                    })
            cv_raw_df = pd.DataFrame(cv_raw_data)
            
            # 计算交叉验证各指标的均值
            cv_mean_data = cv_raw_df.groupby("Subset_Name").agg({
                "Feature_Count": "first",  # 特征数每个子集唯一，取第一个
                "Fold_Val_AUC": "mean",    # CV平均AUC
                "Fold_Val_Recall": "mean",  # CV平均Recall
                "Fold_Val_RMSE": "mean"     # CV平均RMSE
            }).reset_index()
            
            # 重命名列，与绘图逻辑匹配
            cv_mean_data.columns = [
                "Subset", "Feature_Count", 
                "CV_Mean_Val_AUC", "CV_Mean_Val_Recall", "CV_Mean_Val_RMSE"
            ]
            cv_mean_data = cv_mean_data.sort_values(by="CV_Mean_Val_AUC", ascending=False)

            # 使用统一颜色字典
            subset_order = cv_mean_data["Subset"].tolist()

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 13), gridspec_kw={"height_ratios": [1, 1.8]})

            # 上子图：散点 + 误差线
            for subset in subset_order:
                subset_df = cv_raw_df[cv_raw_df["Subset_Name"] == subset]
                mean_auc = subset_df["Fold_Val_AUC"].mean()
                auc_std = subset_df["Fold_Val_AUC"].std()
                feature_count = subset_df["Feature_Count"].iloc[0]  # 每个子集特征数一致

                ax1.scatter(
                    x=feature_count,
                    y=mean_auc,
                    color=self.color_dict[subset],  # 使用统一颜色
                    s=150,
                    edgecolor="black",
                    alpha=0.8,
                    label=subset
                )
                ax1.errorbar(
                    x=feature_count,
                    y=mean_auc,
                    yerr=auc_std,
                    fmt="none",
                    color="gray",
                    capsize=5
                )

            ax1.set_xlabel("Feature Count", fontsize=10)
            ax1.set_ylabel("ROC-AUC", fontsize=10)
            # 标题加粗
            ax1.set_title("Feature Count vs. Cross-Validation Mean Val ROC-AUC", fontsize=11)
            ax1.grid(False)
            # 调整上子图图例（删除标题，与ROC图保持一致）
            ax1.legend(
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                fontsize=8,
                markerscale=0.6
            )

            # 下子图：多指标柱状图
            cv_comparison_long = pd.melt(
                cv_mean_data,
                id_vars="Subset",
                value_vars=["CV_Mean_Val_AUC", "CV_Mean_Val_Recall", "CV_Mean_Val_RMSE"],
                var_name="Metric",
                value_name="Score"
            )
            # 映射指标名称为可读性更强的标签
            metric_mapping = {
                "CV_Mean_Val_AUC": "Mean Val ROC-AUC",
                "CV_Mean_Val_Recall": "Mean Val Recall (TPR)",
                "CV_Mean_Val_RMSE": "Mean Val RMSE"
            }
            cv_comparison_long["Metric"] = cv_comparison_long["Metric"].map(metric_mapping)
            # 按子集顺序排序，确保与上子图一致
            cv_comparison_long["Subset"] = pd.Categorical(
                cv_comparison_long["Subset"],
                categories=subset_order,
                ordered=True
            )

            sns.barplot(
                x="Score",
                y="Subset",
                hue="Metric",
                data=cv_comparison_long,
                palette="Set2",
                ax=ax2
            )
            # 添加数值标签
            for container in ax2.containers:
                ax2.bar_label(container, fmt="%.4f", fontsize=8)
            ax2.set_xlabel("Score", fontsize=10)
            ax2.set_ylabel("Feature Subset", fontsize=10)
            # 标题加粗
            ax2.set_title("Comprehensive Performance Comparison (Cross-Validation)", fontsize=11)
            ax2.legend(
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                fontsize=9
            )

            plt.subplots_adjust(right=0.8)
            
            if save:
                save_path = self.fig_dir / 'rq1_cv_combined_visualization_rf.png'
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"✓ 已保存: {save_path}")
            
            plt.show()
        
    def plot_test_comparison(self, save=True):
        """测试集性能比较图"""
        print("生成测试集性能比较图...")
        
        # 准备数据
        roc_data = pd.DataFrame({
            'Subset': list(self.test_results_detailed.keys()),
            'ROC-AUC': [self.test_results_detailed[k]['Test_ROC_AUC'] for k in self.test_results_detailed],
            'Feature Count': [self.test_results_detailed[k]['Features'] for k in self.test_results_detailed]
        })

        plt.figure(figsize=(12, 6))

        # 子图1: 特征数 vs ROC-AUC
        plt.subplot(1, 2, 1)
        # 使用统一颜色字典
        for subset in roc_data['Subset'].unique():
            subset_data = roc_data[roc_data['Subset'] == subset]
            plt.scatter(
                x=subset_data['Feature Count'],
                y=subset_data['ROC-AUC'],
                color=self.color_dict[subset],
                s=100,
                edgecolor="black",
                label=subset
            )
        # 标题
        plt.title('Feature Count vs. ROC-AUC')
        plt.xlabel('Feature Count')
        plt.ylabel('ROC-AUC')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # 准备综合性能比较数据
        comparison_data = pd.DataFrame({
            'Subset': list(self.test_results_detailed.keys()),
            'ROC-AUC': [self.test_results_detailed[k]['Test_ROC_AUC'] for k in self.test_results_detailed],
            'Recall': [self.test_results_detailed[k]['Recall'] for k in self.test_results_detailed],
            'FPR': [self.test_results_detailed[k]['FPR'] for k in self.test_results_detailed]
        })

        # 转换数据格式用于柱状图
        comparison_data = comparison_data.melt(id_vars='Subset', var_name='Metric', value_name='Score')

        # 子图2: 综合性能比较
        plt.subplot(1, 2, 2)
        sns.barplot(x='Score', y='Subset', hue='Metric', data=comparison_data, palette='Set2')
        # 标题加粗
        plt.title('Comprehensive Performance Comparison')
        plt.xlabel('Score')
        plt.ylabel('Subset')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        
        if save:
            save_path = self.fig_dir / 'rq1_test_performance_comparison_rf.png'
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ 已保存: {save_path}")
        
        plt.show()
        
    def plot_test_roc(self, save=True):
        """测试集ROC曲线比较"""
        print("生成测试集ROC曲线比较...")
        
        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', alpha=0.8)
        
        # 使用统一颜色字典
        for subset_name, result in self.test_results_detailed.items():
            # 使用预存的预测概率
            fpr, tpr, _ = roc_curve(self.y_test, result['y_test_pred_proba'])
            roc_auc = result['roc_auc_test']  # 使用预计算的AUC
            
            plt.plot(
                fpr, tpr, lw=2, 
                label=f'{subset_name} (AUC = {roc_auc:.3f})', 
                color=self.color_dict[subset_name]  # 保持颜色一致
            )
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # 标题
        plt.title('ROC Curve Comparison for Different Feature Subsets(Random Forest)')
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        if save:
            save_path = self.fig_dir / 'rq1_test_roc_rf.png'
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ 已保存: {save_path}")
        
        plt.show()
        
    def plot_all(self):
        """生成所有可视化图表"""
        print("\n" + "="*60)
        print("生成所有可视化图表...")
        print("="*60 + "\n")
        
        self.plot_cv_roc()
        self.plot_cv_combined()
        self.plot_test_comparison()
        self.plot_test_roc()
        
        print("\n" + "="*60)
        print("所有可视化图表生成完成!")
        print(f"   保存路径: {self.fig_dir}")
        print("="*60)


# ===== 主程序入口 =====
if __name__ == '__main__':
    print("\n" + "="*60)
    print("RQ1 可视化脚本")
    print("="*60 + "\n")
    
    # 创建可视化器（使用指定的results目录）
    visualizer = RQ1Visualizer(
        results_dir='/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/Project/results',
        fig_dir='/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/Project/figures'
    )
    
    # 生成所有图表
    visualizer.plot_all()
    
    # 或者单独生成某个图表
    # visualizer.plot_cv_roc()
    # visualizer.plot_cv_combined()
    # visualizer.plot_test_comparison()
    # visualizer.plot_test_roc()
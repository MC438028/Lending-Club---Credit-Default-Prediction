import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import warnings

class RQ2Visualizer:
    
    def __init__(self, results_dir='results', fig_dir='figures'):
        self.results_dir = Path(results_dir)
        self.fig_dir = Path(fig_dir)
        self.fig_dir.mkdir(exist_ok=True)

        self.target_subsets = [
            'LightGBM',
            'WeightedInt_47',
            'WeightedInt_50'
        ]

        self._load_results()
        self._init_style_dict()  # 合并样式定义

    def _load_results(self):
        """Load test results for the three subsets to compare"""
        print("Loading result data...")
        weighted_path = self.results_dir / 'weighted_methods_test_metrics.csv'
        weighted_df = pd.read_csv(weighted_path)
        
        self.test_results_detailed = {}
        
        # Load LightGBM results
        step1_test_path = self.results_dir / 'rq1_test_metrics.csv'
        step1_test_df = pd.read_csv(step1_test_path)
        lgbm_row = step1_test_df[step1_test_df['Subset'] == 'LightGBM'].iloc[0]
        self.test_results_detailed['LightGBM'] = {
            'y_test_pred_proba': lgbm_row['y_test_pred_proba'],
            'roc_auc_test': lgbm_row['Test_ROC_AUC'],
            'y_test': lgbm_row['y_test']
        }
        
        # Load WeightedInt_47
        weighted_47_row = weighted_df[weighted_df['Subset'] == 'WeightedInt_47'].iloc[0]
        self.test_results_detailed['WeightedInt_47'] = {
            'y_test_pred_proba': weighted_47_row['y_test_pred_proba'],
            'roc_auc_test': weighted_47_row['Test_ROC_AUC'],
            'y_test': weighted_47_row['y_test']
        }
        
        # Load WeightedInt_50
        weighted_50_row = weighted_df[weighted_df['Subset'] == 'WeightedInt_50'].iloc[0]
        self.test_results_detailed['WeightedInt_50'] = {
            'y_test_pred_proba': weighted_50_row['y_test_pred_proba'],
            'roc_auc_test': weighted_50_row['Test_ROC_AUC'],
            'y_test': weighted_50_row['y_test']
        }
        
        for subset in self.target_subsets:
            if subset not in self.test_results_detailed:
                raise ValueError(f"Result for {subset} not found. Check data path.")
        
        print(f"✓ Successfully loaded {len(self.test_results_detailed)} subsets")


    def _init_style_dict(self):
        """Define consistent colors, line styles and markers"""
        self.style_dict = {
            'LightGBM': {
                'color': '#1f77b4',      # 蓝色
                'linestyle': (0, (1, 1)), # 点线
                'lw': 2
            },
            'WeightedInt_47': {
                'color': '#d62728',      # 红色
                'linestyle': (0, (5, 2)), # 虚线
                'lw': 2
            },
            'WeightedInt_50': {
                'color': '#2ca02c',      # 绿色
                'linestyle': '-',         # 实线
                'lw': 2
            }
        }


    def plot_comparison_roc(self, save_path=None):
        """Plot ROC curve with inset above legend and smaller zoomed area"""
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        
        # Plot main curves (only main plot has legend labels)
        ax.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='gray', label="Chance")
        for subset_name in self.target_subsets:
            result = self.test_results_detailed[subset_name]
            y_pred_proba = eval(result['y_test_pred_proba']) if isinstance(result['y_test_pred_proba'], str) else result['y_test_pred_proba']
            y_test = eval(result['y_test']) if isinstance(result['y_test'], str) else result['y_test']
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = result['roc_auc_test']
            style = self.style_dict[subset_name]
            
            ax.plot(
                fpr, tpr,
                label=f'{subset_name} (AUC = {roc_auc:.4f})',
                color=style['color'],
                linestyle=style['linestyle'],
                lw=style['lw']
            )

        # Main plot settings
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title('ROC Curve Comparison: Weighted Integrated vs. LightGBM-based Subsets',
                  fontsize=12, pad=10)
        ax.legend(loc='lower right', fontsize=9, frameon=True, framealpha=0.9)  # Legend at lower right
        ax.grid(False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Inset above legend (adjust loc and borderpad to position correctly)
        axins = inset_axes(
            ax, 
            width="30%",  # Smaller inset width
            height="30%", # Smaller inset height
            loc='center right',  # Position above lower-right legend
            borderpad=1.2  # Adjust distance from main plot edges
        )

        # Plot inset curves (no labels to avoid duplicate legend)
        for subset_name in self.target_subsets:
            result = self.test_results_detailed[subset_name]
            y_pred_proba = eval(result['y_test_pred_proba']) if isinstance(result['y_test_pred_proba'], str) else result['y_test_pred_proba']
            y_test = eval(result['y_test']) if isinstance(result['y_test'], str) else result['y_test']
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            style = self.style_dict[subset_name]
            axins.plot(fpr, tpr, **style)

        # Smaller zoom area (narrower range to focus on differences)
        axins.set_xlim(0.32, 0.44)  # Narrower x range
        axins.set_ylim(0.69, 0.81)  # Narrower y range
        axins.grid(alpha=0.3)
        axins.tick_params(labelsize=7)  # Smaller tick labels

        

        # Connect inset to main plot with subtle lines
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.3", lw=0.8, linestyle='--')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=400, bbox_inches="tight")
            print(f"✓ Saved to: {save_path}")

        plt.show()


    def plot_all(self):
        save_path = self.fig_dir / "ROC_optimized_inset.png"
        self.plot_comparison_roc(save_path=save_path)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Weighted Subset Visualization")
    print("="*60)
    
    visualizer = RQ2Visualizer(
        results_dir='/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/Project/results',
        fig_dir='/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/Project/figures'
    )
    visualizer.plot_all()
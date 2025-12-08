import pickle
import pandas as pd
from pathlib import Path

class FinalFeatureGenerator:
    def __init__(self, 
                 feature_file="data/selected_features.pkl",
                 weights_file="results/rq2_optimized_weights_rf.pkl",
                 output_dir="data"):  
        self.feature_file = Path(feature_file)
        self.weights_file = Path(weights_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self._load_data()
    
    def _load_data(self):
        """加载特征子集和优化的权重（适配step2的after_removal和after_replacement）"""
        print("="*60)
        print("加载特征和权重数据...")
        
        # 1. 加载特征子集
        with open(self.feature_file, "rb") as f:
            self.all_feature_dict = pickle.load(f)
        print(f"✓ 加载所有特征子集: {list(self.all_feature_dict.keys())}")

        # 2. 加载优化结果（对应step2的三种处理方式）
        with open(self.weights_file, "rb") as f:
            self.optimization_results = pickle.load(f)
        
        # 验证结果格式
        required_keys = ['original']
        for key in required_keys:
            if key not in self.optimization_results:
                raise ValueError(f"权重文件缺少必要的 {key} 数据，请检查step2输出")

        # 提取方法一（原始权重）的关键信息
        self.results = {
            'original': {  # 方法一：原始权重
                'subset_weights': self.optimization_results['original']['subset_weights'],
                'best_n_features': self.optimization_results['original']['weighted_feature_count']
            }
            
        }
        
        # 打印每个方法的单个子集权重（带占比）
        for method in ['original']:
            print(f"\n{method}（原始）运用到的单个子集权重:")
            print("-"*40)
            total_weight = sum(self.results[method]['subset_weights'].values())
            for subset, weight in self.results[method]['subset_weights'].items():
                print(f"  {subset}: {weight:.4f} ({weight/total_weight:.2%})")
            print(f"✓ {method}最佳特征数量: {self.results[method]['best_n_features']}")
        
        # 3. 过滤出有效子集并预计算特征得分
        self.feature_scores = {}
        self.sorted_features = {}
        
        for method in ['original']:
            # 过滤有效子集
            valid_subsets = list(self.results[method]['subset_weights'].keys())
            feature_dict = {
                subset: self.all_feature_dict[subset] 
                for subset in valid_subsets 
                if subset in self.all_feature_dict
            }
            
            # 校验子集完整性
            missing_subsets = [s for s in valid_subsets if s not in feature_dict]
            if missing_subsets:
                raise ValueError(f"特征数据中缺少{method}的以下子集: {missing_subsets}")
            print(f"✓ {method}有效特征子集: {list(feature_dict.keys())}")
            
            # 预计算特征得分
            self._calculate_feature_scores(method, feature_dict, self.results[method]['subset_weights'])
    
    def _calculate_feature_scores(self, method, feature_dict, subset_weights):
        """为指定的方法计算特征得分"""
        # 收集所有候选特征（去重）
        all_features = set()
        for features in feature_dict.values():
            all_features.update(features)
        all_features = list(all_features)
        
        # 计算每个特征的加权得分（累加所在子集的权重）
        feature_scores = {feat: 0.0 for feat in all_features}
        for subset, weight in subset_weights.items():
            for feat in feature_dict[subset]:
                feature_scores[feat] += weight
        
        # 按得分降序排序
        self.sorted_features[method] = sorted(
            feature_scores.items(), 
            key=lambda x: (-x[1], x[0])  # 先按得分降序，再按特征名升序
        )
        print(f"✓ {method}特征得分计算完成（共 {len(self.sorted_features[method])} 个特征）")
    
    def generate_features(self, n_features, method):
        """生成指定数量的特征并保存（子集名称固定为WeightedInt_+特征数）"""
        subset_name = f"WeightedInt_{n_features}"  # 严格按照要求命名
        
        print("\n" + "="*60)
        print(f"基于{method}生成 {subset_name}（{n_features} 个特征）...")
        
        # 选取指定数量的特征
        selected_features = [feat for feat, score in self.sorted_features[method][:n_features]]
        print(f"✓ 已选择 {len(selected_features)} 个特征（目标: {n_features}）")
        
        # 保存结果（pkl文件名自定义，区分方法和特征数）
        file_name = f"{method}_features_{n_features}.pkl"
        self._save_features(selected_features, self.sorted_features[method], subset_name, method, file_name)
        
        return selected_features, subset_name
    
    def _save_features(self, selected_features, all_sorted, subset_name, method, file_name):
        """保存特征集及得分详情"""
        # 保存特征列表（pkl文件）
        feat_path = self.output_dir / file_name
        with open(feat_path, "wb") as f:
            pickle.dump({
                "subset_name": subset_name,
                "features": selected_features
            }, f)
        print(f"✓ 特征列表已保存: {feat_path}")
        
        # 保存特征得分详情
        scores_df = pd.DataFrame(all_sorted, columns=["Feature", "Weighted_Score"])
        scores_df["From_Subsets"] = scores_df["Feature"].apply(
            lambda x: [s for s in self.results[method]['subset_weights'].keys() 
                      if x in self.all_feature_dict.get(s, [])]
        )
        scores_df.to_csv(self.output_dir / f"{subset_name}_scores.csv", index=False)
        print(f"✓ 特征得分详情已保存: {self.output_dir / f'{subset_name}_scores.csv'}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Step3: 为方法二（after_removal）和方法三（after_replacement）生成特征集文件")
    print("="*60)
    
    generator = FinalFeatureGenerator(
        feature_file="/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/Project/data/selected_features.pkl",
        weights_file="/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/Project/results/rq2_optimized_weights_rf.pkl",  # 对应step2的输出文件
        output_dir="/Users/xumoyan/Program/anaconda3/envs/cisc7201/final report/ML/Project/data"
    )
    
    # 生成特征集
    print("\n" + "="*80)
    print("===== 开始生成特征集 =====")
    print("="*80)
    # 最佳数量特征集
    best_count = generator.results['original']['best_n_features']
    best_features, best_name = generator.generate_features(
        n_features=best_count,
        method='original'
    )
    
    
    # 输出结果总结
    print("\n" + "="*80)
    print("特征集生成完成，文件清单：")
    print("="*80)
    print(f"1. {best_name}: original_features_{best_count}.pkl ({len(best_features)}个特征)")
    print(f"\n所有文件均保存至: {generator.output_dir}")
    print("="*80)
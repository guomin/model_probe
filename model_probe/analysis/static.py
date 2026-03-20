"""
静态分析模块：直接分析模型权重/参数
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np


class WeightAnalyzer:
    """
    权重分析器：分析模型参数的统计特性
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        
    def compute_weight_statistics(self) -> Dict[str, Dict[str, float]]:
        """计算各层权重的统计信息"""
        stats = {}
        
        for name, param in self.model.named_parameters():
            if "weight" in name:
                data = param.data.flatten().cpu().numpy()
                stats[name] = {
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data)),
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                    "norm": float(np.linalg.norm(data)),
                    "sparsity": float(np.mean(np.abs(data) < 1e-4))
                }
                
        return stats
    
    def compute_svd(self, layer_name: str, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算指定层的 SVD 分解
        
        Returns:
            (奇异值, 解释方差比例)
        """
        param = None
        for name, p in self.model.named_parameters():
            if layer_name in name and "weight" in name:
                param = p.data
                break
                
        if param is None:
            raise ValueError(f"Layer {layer_name} not found")
            
        U, S, V = torch.svd(param)
        singular_values = S.cpu().numpy()[:k]

        variance_ratio = (S[:k] ** 2) / (S ** 2).sum().item()
        variance_ratio = variance_ratio.cpu().numpy()

        return singular_values, variance_ratio
    
    def compute_rank(self, layer_name: str, threshold: float = 1e-6) -> int:
        """计算权重矩阵的有效秩"""
        param = None
        for name, p in self.model.named_parameters():
            if layer_name in name and "weight" in name:
                param = p.data
                break
                
        if param is None:
            raise ValueError(f"Layer {layer_name} not found")
            
        S = torch.linalg.svdvals(param)
        full_rank = S.shape[0]
        effective_rank = (S > threshold * S[0]).sum().item()
        
        return int(effective_rank)
    
    def analyze_attention_patterns_static(self, layer_name: str) -> Dict[str, float]:
        """分析 Attention 层的权重模式"""
        for name, param in self.model.named_parameters():
            if layer_name in name and "weight" in name:
                weight = param.data
                
                if weight.dim() == 2 and weight.shape[0] == weight.shape[1]:
                    eigvals = torch.linalg.eigvals(weight)
                    eigvals_real = eigvals.real
                    
                    return {
                        "mean": weight.mean().item(),
                        "std": weight.std().item(),
                        "max_eig": eigvals_real.max().item(),
                        "min_eig": eigvals_real.min().item(),
                        "condition_number": (eigvals_real.max() / (eigvals_real.min() + 1e-8)).item()
                    }
                    
        return {}
    
    def compute_parameter_count(self) -> Dict[str, int]:
        """计算各层参数量"""
        counts = {}
        total = 0
        
        for name, param in self.model.named_parameters():
            count = param.numel()
            counts[name] = count
            total += count
            
        counts["_total"] = total
        return counts
    
    def compute_layer_similarity(self, metric: str = "cosine") -> Dict[Tuple[str, str], float]:
        """
        计算模型各层之间的相似性
        
        Args:
            metric: "cosine" (余弦相似度) 或 "cka" (Centered Kernel Alignment)
            
        Returns:
            {(layer1, layer2): similarity_score}
        """
        if metric == "cosine":
            return self._cosine_similarity_between_layers()
        elif metric == "cka":
            return self._cka_similarity_between_layers()
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def _cosine_similarity_between_layers(self) -> Dict[Tuple[str, str], float]:
        """计算各层权重的余弦相似度"""
        layers = []
        weights = []
        
        for name, param in self.model.named_parameters():
            if "weight" in name:
                layers.append(name)
                weights.append(param.data.flatten().float())
        
        n = len(layers)
        similarity = {}
        
        for i in range(n):
            for j in range(i + 1, n):
                w1 = weights[i]
                w2 = weights[j]
                min_len = min(len(w1), len(w2))
                sim = torch.nn.functional.cosine_similarity(
                    w1[:min_len], w2[:min_len], dim=0
                ).item()
                similarity[(layers[i], layers[j])] = sim
                
        return similarity
    
    def _cka_similarity_between_layers(self) -> Dict[Tuple[str, str], float]:
        """
        计算各层之间的 CKA (Centered Kernel Alignment) 相似度
        """
        layers = []
        weights = []
        
        for name, param in self.model.named_parameters():
            if "weight" in name:
                layers.append(name)
                w = param.data.float()
                if w.dim() > 1:
                    w = w.reshape(w.shape[0], -1)
                else:
                    w = w.unsqueeze(0)
                weights.append(w.T)
        
        n = len(layers)
        similarity = {}
        
        def centering(K):
            n = K.shape[0]
            I = torch.eye(n, device=K.device)
            H = I - torch.ones(n, n, device=K.device) / n
            return H @ K @ H
        
        def rbf_kernel(X, sigma=1.0):
            X_norm = (X ** 2).sum(dim=1).reshape(-1, 1)
            dist = X_norm + X_norm.T - 2 * X @ X.T
            return torch.exp(-dist / (2 * sigma ** 2))
        
        def cka(K1, K2):
            K1 = centering(K1)
            K2 = centering(K2)
            hsic = (K1 * K2).sum()
            var1 = (K1 ** 2).sum()
            var2 = (K2 ** 2).sum()
            return hsic / (torch.sqrt(var1 * var2) + 1e-10)
        
        for i in range(n):
            for j in range(i + 1, n):
                K1 = rbf_kernel(weights[i])
                K2 = rbf_kernel(weights[j])
                sim = cka(K1, K2).item()
                similarity[(layers[i], layers[j])] = sim
                
        return similarity
    
    def visualize_similarity_matrix(self, metric: str = "cosine", save_path: Optional[str] = None):
        """生成层相似性热力图"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        sim_dict = self.compute_layer_similarity(metric)
        layers = list(set([l for pair in sim_dict.keys() for l in pair]))
        layers.sort()
        
        n = len(layers)
        sim_matrix = torch.zeros(n, n)
        idx = {l: i for i, l in enumerate(layers)}
        
        for (l1, l2), sim in sim_dict.items():
            i, j = idx[l1], idx[l2]
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim
        sim_matrix.fill_diagonal_(1.0)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(sim_matrix.numpy(), 
                    xticklabels=layers, 
                    yticklabels=layers,
                    annot=False, 
                    fmt='.2f',
                    cmap='RdYlBu_r',
                    vmin=-1, vmax=1)
        plt.title(f'Layer Similarity ({metric.upper()})')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        return sim_matrix, layers

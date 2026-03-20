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
        
        variance_ratio = (singular_values ** 2) / (S ** 2).sum().item()
        variance_ratio = variance_ratio.cpu().numpy()[:k]
        
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

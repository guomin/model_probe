"""
分析模块：激活分析、归因分析
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


class ActivationAnalyzer:
    """
    激活分析器：分析神经元和层的激活模式
    """
    
    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper
        
    def compute_layer_statistics(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        计算各层的激活统计信息
        
        Returns:
            {layer_name: {mean, std, min, max, sparsity}}
        """
        hidden_states = self.model_wrapper.get_hidden_states(
            input_ids, attention_mask
        )["all_hidden_states"]
        
        stats = {}
        
        for i, hidden in enumerate(hidden_states):
            layer_name = f"layer_{i}"
            hidden_flat = hidden.flatten()
            
            stats[layer_name] = {
                "mean": hidden_flat.mean().item(),
                "std": hidden_flat.std().item(),
                "min": hidden_flat.min().item(),
                "max": hidden_flat.max().item(),
                "sparsity": (hidden_flat.abs() < 1e-4).float().mean().item(),
            }
            
        return stats
    
    def find_important_neurons(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        top_k: int = 100
    ) -> Dict[str, List[int]]:
        """
        找出激活值最高的神经元
        
        Returns:
            {layer_name: [neuron_indices]}
        """
        hidden_states = self.model_wrapper.get_hidden_states(
            input_ids, attention_mask
        )["all_hidden_states"]
        
        important_neurons = {}
        
        for i, hidden in enumerate(hidden_states):
            layer_name = f"layer_{i}"
            mean_activations = hidden.mean(dim=(0, 1))
            top_indices = mean_activations.argsort(descending=True)[:top_k]
            important_neurons[layer_name] = top_indices.tolist()
            
        return important_neurons
    
    def analyze_attention_patterns(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, np.ndarray]:
        """
        分析注意力模式
        
        Returns:
            {head_name: attention_matrix}
        """
        mask = attention_mask.to(self.model_wrapper.config.device) if attention_mask is not None else None
        outputs = self.model_wrapper.model(
            input_ids=input_ids.to(self.model_wrapper.config.device),
            attention_mask=mask,
            output_attentions=True,
            use_cache=False
        )
        
        attentions = outputs.attentions
        attention_patterns = {}
        
        for layer_idx, attn in enumerate(attentions):
            attn = attn[0]
            num_heads = attn.shape[0]
            
            for head_idx in range(num_heads):
                head_name = f"layer{layer_idx}_head{head_idx}"
                attention_patterns[head_name] = attn[head_idx].detach().cpu().numpy()
                
        return attention_patterns
    
    def visualize_attention(
        self,
        input_ids: torch.Tensor,
        tokens: List[str],
        layer_idx: int = 0,
        head_idx: int = 0,
        output_path: str = "attention.png"
    ):
        """可视化注意力权重"""
        outputs = self.model_wrapper.model(
            input_ids=input_ids.to(self.model_wrapper.config.device),
            output_attentions=True,
            use_cache=False
        )
        
        attn = outputs.attentions[layer_idx][0, head_idx].cpu().numpy()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            attn[:len(tokens), :len(tokens)],
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="viridis",
            square=True
        )
        plt.title(f"Attention Layer {layer_idx} Head {head_idx}")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"Saved attention visualization to {output_path}")


class Attributor:
    """
    归因分析器：计算输入对输出的贡献
    """
    
    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper
        
    def integrated_gradients(
        self,
        input_ids: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        target_idx: Optional[int] = None,
        steps: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrated Gradients 归因
        
        Args:
            input_ids: 输入token ids
            baseline: 基线输入（默认为零向量）
            target_idx: 目标token位置
            steps: 积分步数
            
        Returns:
            (归因分数, 预测logits)
        """
        if baseline is None:
            baseline = torch.zeros_like(input_ids)
            
        input_ids = input_ids.to(self.model_wrapper.config.device)
        baseline = baseline.to(self.model_wrapper.config.device)
        
        input_embed = self.model_wrapper.model.get_input_embeddings()
        
        input_embedding = input_embed(input_ids)
        baseline_embedding = input_embed(baseline)
        
        gradients = []
        
        for alpha in torch.linspace(0, 1, steps):
            interpolated = baseline_embedding + alpha * (input_embedding - baseline_embedding)
            interpolated = interpolated.requires_grad_(True)
            
            outputs = self.model_wrapper.model(inputs_embeds=interpolated, use_cache=False)
            logits = outputs.logits[0]
            
            if target_idx is None:
                target_idx = logits.argmax(-1)
                
            self.model_wrapper.model.zero_grad()
            logits[target_idx].backward()
            
            gradients.append(interpolated.grad.detach())
            
        gradients = torch.stack(gradients)
        avg_gradients = gradients.mean(dim=0)
        
        ig_scores = (input_embedding - baseline_embedding) * avg_gradients
        
        return ig_scores, logits
    
    def gradient_x_input(
        self,
        input_ids: torch.Tensor,
        target_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gradient × Input 归因
        """
        input_ids = input_ids.to(self.model_wrapper.config.device)
        input_ids.requires_grad_(True)
        
        outputs = self.model_wrapper.model(input_ids, use_cache=False)
        logits = outputs.logits[0]
        
        if target_idx is None:
            target_idx = logits.argmax(-1)
            
        self.model_wrapper.model.zero_grad()
        logits[target_idx].backward()
        
        input_grads = input_ids.grad
        input_values = input_ids.float()
        
        attribution = input_grads * input_values
        
        return attribution, logits
    
    def attention_rollout(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Attention Rollout：计算token之间的完整注意力连接
        """
        outputs = self.model_wrapper.model(
            input_ids=input_ids.to(self.model_wrapper.config.device),
            attention_mask=attention_mask.to(self.model_wrapper.config.device) if attention_mask else None,
            output_attentions=True,
            use_cache=False
        )
        
        attentions = outputs.attentions
        
        num_layers = len(attentions)
        num_heads = attentions[0].shape[1]
        
        rollout = attentions[0].mean(dim=1)
        
        for i in range(1, num_layers):
            attn = attentions[i].mean(dim=1)
            
            rollout = rollout @ (attn + torch.eye(attn.shape[-1]).to(attn.device)) / (num_heads + 1)
            
        return rollout[0].cpu().numpy()
    
    def compute_layerwise_importance(
        self,
        input_ids: torch.Tensor,
        target_idx: int = -1
    ) -> Dict[str, float]:
        """
        计算每层对最终输出的重要性（基于激活幅度）
        """
        input_ids = input_ids.to(self.model_wrapper.config.device)
        
        with torch.no_grad():
            outputs = self.model_wrapper.model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False
            )
        
        hidden_states = outputs.hidden_states
        
        layer_importance = {}
        
        for i, hidden in enumerate(hidden_states):
            importance = hidden.abs().mean().item()
            layer_importance[f"layer_{i}"] = importance
            
        return layer_importance

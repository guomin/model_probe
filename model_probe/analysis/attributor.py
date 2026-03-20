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
    
    def compute_layer_activation_similarity(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        metric: str = "cka"
    ) -> Dict[Tuple[str, str], float]:
        """
        动态分析：计算各层激活值之间的相似性
        
        Args:
            input_ids: 输入token ids
            attention_mask: 注意力掩码
            metric: "cosine" 或 "cka"
            
        Returns:
            {(layer_i, layer_j): similarity}
        """
        hidden_states = self.model_wrapper.get_hidden_states(
            input_ids, attention_mask
        )["all_hidden_states"]
        
        n_layers = len(hidden_states)
        activations = []
        layer_names = []
        
        for i, hidden in enumerate(hidden_states):
            hidden_mean = hidden.mean(dim=1)
            activations.append(hidden_mean.flatten(1).T)
            layer_names.append(f"layer_{i}")
        
        similarity = {}
        
        if metric == "cosine":
            for i in range(n_layers):
                for j in range(i + 1, n_layers):
                    sim = F.cosine_similarity(activations[i], activations[j]).mean().item()
                    similarity[(layer_names[i], layer_names[j])] = sim
        else:
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
            
            for i in range(n_layers):
                for j in range(i + 1, n_layers):
                    K1 = rbf_kernel(activations[i])
                    K2 = rbf_kernel(activations[j])
                    sim = cka(K1, K2).item()
                    similarity[(layer_names[i], layer_names[j])] = sim
        
        return similarity

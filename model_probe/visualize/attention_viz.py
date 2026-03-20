"""
可视化模块：生成人类可理解的图表
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict


class AttentionVisualizer:
    """注意力可视化"""
    
    def plot_attention_head(
        self,
        attention: np.ndarray,
        tokens: List[str],
        output_path: str,
        title: str = "Attention Pattern"
    ):
        """绘制单个注意力头"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            attention[:len(tokens), :len(tokens)],
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="viridis",
            square=True
        )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
    def plot_attention_grid(
        self,
        attentions: Dict[str, np.ndarray],
        tokens: List[str],
        output_path: str,
        max_heads: int = 16
    ):
        """绘制所有注意力头的网格"""
        n_heads = min(len(attentions), max_heads)
        cols = 4
        rows = (n_heads + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        axes = axes.flatten() if rows > 1 else [axes] if n_heads == 1 else []
        
        for idx, (name, attn) in enumerate(list(attentions.items())[:max_heads]):
            ax = axes[idx]
            sns.heatmap(
                attn[:len(tokens), :len(tokens)],
                xticklabels=[],
                yticklabels=[],
                cmap="viridis",
                ax=ax,
                square=True
            )
            ax.set_title(name, fontsize=8)
            
        for idx in range(len(attentions), len(axes)):
            axes[idx].axis("off")
            
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


class LayerVisualizer:
    """层信息可视化"""
    
    def plot_layer_statistics(
        self,
        stats: Dict[str, Dict[str, float]],
        output_path: str
    ):
        """绘制层统计信息"""
        layer_names = list(stats.keys())
        means = [stats[l]["mean"] for l in layer_names]
        stds = [stats[l]["std"] for l in layer_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.bar(layer_names, means)
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Mean")
        ax1.set_title("Layer Mean Activations")
        ax1.tick_params(axis="x", rotation=45)
        
        ax2.bar(layer_names, stds)
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Std")
        ax2.set_title("Layer Std Deviation")
        ax2.tick_params(axis="x", rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
    def plot_layer_similarity(
        self,
        similarity: np.ndarray,
        layer_names: List[str],
        output_path: str
    ):
        """绘制层相似度热力图"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            similarity,
            xticklabels=layer_names,
            yticklabels=layer_names,
            cmap="coolwarm",
            center=0,
            annot=True,
            fmt=".2f"
        )
        plt.title("Layer Representation Similarity")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


class ProbeVisualizer:
    """探针结果可视化"""
    
    def plot_layer_accuracy(
        self,
        layer_accuracies: Dict[str, float],
        output_path: str
    ):
        """绘制各层探针准确率"""
        layers = list(layer_accuracies.keys())
        accuracies = list(layer_accuracies.values())
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(layers, accuracies, color="steelblue")
        
        best_idx = accuracies.index(max(accuracies))
        bars[best_idx].set_color("coral")
        
        plt.xlabel("Layer")
        plt.ylabel("Accuracy")
        plt.title("Probe Accuracy by Layer")
        plt.xticks(rotation=45)
        plt.ylim(0, 1.1)
        
        for i, acc in enumerate(accuracies):
            plt.text(i, acc + 0.02, f"{acc:.2f}", ha="center", fontsize=8)
            
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

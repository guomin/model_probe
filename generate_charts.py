#!/usr/bin/env python3
"""
生成可视化图表
"""

import sys
sys.path.insert(0, "/mnt/hdsd1/guomin/projects/model_probe")

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model_probe.core import ModelWrapper, ModelConfig
from model_probe.probes import LinearProbe, ProbeConfig
from model_probe.analysis import ActivationAnalyzer

plt.rcParams['font.size'] = 10
plt.rcParams['figure.facecolor'] = 'white'

# ========== 数据准备 ==========

# 语法知识结果
grammar_layers = list(range(13))
grammar_scores = [90.48] * 13

# 实体知识结果  
entity_layers = list(range(13))
entity_scores = [23.33, 88.89, 82.22, 83.33, 88.89, 88.89, 88.89, 88.89, 
                 82.22, 82.22, 88.89, 88.89, 94.44]

# 数字推理结果
number_layers = list(range(13))
number_scores = [61.11] + [52.78] * 12

# ========== 图1: 各任务各层准确率对比 ==========

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 语法知识
ax1 = axes[0]
colors1 = plt.cm.Blues(np.linspace(0.3, 0.9, 13))
ax1.bar(grammar_layers, grammar_scores, color=colors1)
ax1.set_xlabel('Layer')
ax1.set_ylabel('CV Accuracy (%)')
ax1.set_title('Task 1: Grammar (Verb vs Gerund)')
ax1.set_ylim(0, 100)
ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random')
ax1.legend()

# 实体知识
ax2 = axes[1]
colors2 = plt.cm.Greens(np.linspace(0.3, 0.9, 13))
ax2.bar(entity_layers, entity_scores, color=colors2)
ax2.set_xlabel('Layer')
ax2.set_ylabel('CV Accuracy (%)')
ax2.set_title('Task 2: Entity Type (Person vs Place)')
ax2.set_ylim(0, 100)
ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random')
ax2.legend()

# 数字推理
ax3 = axes[2]
colors3 = plt.cm.Oranges(np.linspace(0.3, 0.9, 13))
ax3.bar(number_layers, number_scores, color=colors3)
ax3.set_xlabel('Layer')
ax3.set_ylabel('CV Accuracy (%)')
ax3.set_title('Task 3: Number Comparison')
ax3.set_ylim(0, 100)
ax3.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random')
ax3.legend()

plt.tight_layout()
plt.savefig('layer_accuracy_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("Saved: layer_accuracy_comparison.png")

# ========== 图2: 热力图 - 综合对比 ==========

fig, ax = plt.subplots(figsize=(12, 4))

data = np.array([
    grammar_scores,
    entity_scores,
    number_scores
])

sns.heatmap(data, 
            annot=True, 
            fmt='.1f', 
            cmap='RdYlGn',
            vmin=20, vmax=100,
            xticklabels=[f'L{i}' for i in range(13)],
            yticklabels=['Grammar', 'Entity', 'Number'],
            ax=ax)

ax.set_xlabel('Layer')
ax.set_title('Probe Accuracy Across Layers and Tasks')

plt.tight_layout()
plt.savefig('heatmap_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("Saved: heatmap_comparison.png")

# ========== 图3: 层表示演化示意图 ==========

fig, ax = plt.subplots(figsize=(14, 6))

# 绘制三层区域
regions = [
    (0, 3, 'Shallow Layers\n(Basic Features)', '#FFE0B2'),
    (3, 9, 'Middle Layers\n(Syntax & Patterns)', '#C8E6C9'),
    (9, 13, 'Deep Layers\n(Semantics)', '#BBDEFB')
]

for start, end, label, color in regions:
    ax.axvspan(start-0.5, end-0.5, alpha=0.3, color=color)
    ax.text((start+end)/2, 105, label, ha='center', fontsize=11, fontweight='bold')

# 绘制三条曲线
ax.plot(grammar_layers, grammar_scores, 'o-', label='Grammar', linewidth=2, markersize=6, color='blue')
ax.plot(entity_layers, entity_scores, 's-', label='Entity Type', linewidth=2, markersize=6, color='green')
ax.plot(number_layers, number_scores, '^-', label='Number', linewidth=2, markersize=6, color='orange')

ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('Probe Accuracy (%)', fontsize=12)
ax.set_title('GPT-2 Representation Evolution Across Layers', fontsize=14, fontweight='bold')
ax.set_xticks(range(13))
ax.set_xlim(-0.5, 12.5)
ax.set_ylim(0, 110)
ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.text(12.2, 52, 'Random', fontsize=9, color='red')
ax.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('layer_evolution.png', dpi=150, bbox_inches='tight')
plt.close()

print("Saved: layer_evolution.png")

# ========== 图4: 注意力头模式 ==========

config = ModelConfig(name_or_path="gpt2", device="cpu", dtype=torch.float32)
wrapper = ModelWrapper(config)
wrapper.load_model(for_generation=False)

analyzer = ActivationAnalyzer(wrapper)

sentences = [
    "The cat sat on the mat because it was tired.",
    "Paris is the capital of France."
]

fig, axes = plt.subplots(2, 6, figsize=(18, 6))

for row, sent in enumerate(sentences):
    inputs = wrapper.tokenizer(sent, return_tensors="pt", padding=True, truncation=True)
    tokens = wrapper.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    attention_patterns = analyzer.analyze_attention_patterns(
        inputs["input_ids"],
        inputs["attention_mask"]
    )
    
    for head_idx in range(6):
        ax = axes[row, head_idx]
        layer_name = f"layer0_head{head_idx}"
        
        if layer_name in attention_patterns:
            attn = attention_patterns[layer_name]
            n = min(len(tokens), attn.shape[0])
            attn_plot = attn[:n, :n]
            
            sns.heatmap(attn_plot, 
                       xticklabels=tokens[:n] if n <= 8 else False,
                       yticklabels=tokens[:n] if n <= 8 else False,
                       ax=ax, cmap="viridis", cbar=False)
            
        ax.set_title(f"Head {head_idx}", fontsize=9)

plt.suptitle('Attention Patterns: First 6 Heads', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('attention_heads.png', dpi=150, bbox_inches='tight')
plt.close()

print("Saved: attention_heads.png")

# ========== 统计摘要 ==========

print("\n" + "="*60)
print("可视化图表生成完成!")
print("="*60)
print("""
生成的文件:
1. layer_accuracy_comparison.png  - 各任务各层准确率柱状图
2. heatmap_comparison.png         - 准确率热力图
3. layer_evolution.png            - 层表示演化曲线
4. attention_heads.png            - 注意力头模式

核心发现:
- 语法知识: 各层均匀分布 (~90%)
- 实体知识: 深层最好 (Layer 12: 94%)
- 数字推理: 几乎无效 (~50%)
""")

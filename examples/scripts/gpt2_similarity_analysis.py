#!/usr/bin/env python3
"""
GPT-2 模型层相似度分析

分析GPT-2各层之间的相似性，发现模型内部结构
"""

import sys
sys.path.insert(0, "/mnt/hdsd1/guomin/projects/model_probe")

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model_probe import ModelWrapper, ModelConfig
from model_probe.analysis import WeightAnalyzer

print("=" * 80)
print("GPT-2 层相似度分析")
print("=" * 80)

# ============================================
# 1. 加载模型
# ============================================
print("\n【步骤1】加载 GPT-2 模型")
print("-" * 80)

config = ModelConfig(
    name_or_path="gpt2",
    device="cpu"
)

wrapper = ModelWrapper(config)
wrapper.load_model(for_generation=False)

model = wrapper.model

print(f"✓ 模型加载成功")
print(f"  模型: {config.name_or_path}")
print(f"  总参数量: {wrapper.get_parameter_count():,}")

# ============================================
# 2. 创建分析器
# ============================================
print("\n【步骤2】创建相似度分析器")
print("-" * 80)

analyzer = WeightAnalyzer(model)
print("✓ 分析器创建完成")

# ============================================
# 3. 余弦相似度分析
# ============================================
print("\n【步骤3】余弦相似度分析")
print("-" * 80)

print("正在计算余弦相似度（这可能需要几分钟）...")

cosine_sim = analyzer.compute_layer_similarity(metric="cosine")

print(f"✓ 计算完成！共分析了 {len(cosine_sim)} 对层")

# 找出最相似和最不相似的层对
sorted_cosine = sorted(cosine_sim.items(), key=lambda x: abs(x[1]), reverse=True)

print(f"\n最相似的10对层（余弦相似度）：")
print(f"{'排名':<6} {'层1':<50} {'层2':<50} {'相似度':>10}")
print("-" * 130)

for i, ((layer1, layer2), sim) in enumerate(sorted_cosine[:10], 1):
    # 简化层名显示
    l1_short = layer1[:48]
    l2_short = layer2[:48]
    print(f"{i:<6} {l1_short:<50} {l2_short:<50} {abs(sim):>10.4f}")

# ============================================
# 4. 分析Transformer层之间的关系
# ============================================
print("\n【步骤4】Transformer层之间的关系分析")
print("-" * 80)

# 提取所有Transformer层的权重
transformer_layers = []
layer_pairs = []

for name, param in model.named_parameters():
    if 'h.' in name and 'attn.c_attn.weight' in name:
        # 提取层号
        layer_num = int(name.split('.')[1])
        transformer_layers.append((layer_num, name, param))

# 按层号排序
transformer_layers.sort()

print(f"找到 {len(transformer_layers)} 个Transformer注意力层")

# 分析相邻层的相似度
print(f"\n相邻Transformer层的相似度：")
print(f"{'层对':<20} {'余弦相似度':>15} {'趋势':>20}")
print("-" * 60)

adjacent_similarities = []
for i in range(len(transformer_layers) - 1):
    layer1_num, layer1_name, layer1_weight = transformer_layers[i]
    layer2_num, layer2_name, layer2_weight = transformer_layers[i + 1]

    # 计算相似度
    w1 = layer1_weight.data.flatten().float()
    w2 = layer2_weight.data.flatten().float()
    min_len = min(len(w1), len(w2))
    sim = torch.nn.functional.cosine_similarity(w1[:min_len], w2[:min_len], dim=0).item()

    adjacent_similarities.append(sim)

    # 判断趋势
    if abs(sim) > 0.8:
        trend = "高度相似"
    elif abs(sim) > 0.5:
        trend = "较为相似"
    elif abs(sim) > 0.3:
        trend = "中等相似"
    else:
        trend = "差异较大"

    print(f"Layer {layer1_num:2d} -> {layer2_num:2d}  {sim:>14.4f}      {trend:>20}")

# 统计分析
mean_sim = np.mean([abs(s) for s in adjacent_similarities])
std_sim = np.std([abs(s) for s in adjacent_similarities])

print(f"\n统计分析：")
print(f"  平均相似度: {mean_sim:.4f}")
print(f"  标准差: {std_sim:.4f}")
print(f"  最大相似度: {max([abs(s) for s in adjacent_similarities]):.4f}")
print(f"  最小相似度: {min([abs(s) for s in adjacent_similarities]):.4f}")

# ============================================
# 5. 发现模式
# ============================================
print("\n【步骤5】发现模式")
print("-" * 80)

# 按相似度分组
high_sim_pairs = [(i, abs(adjacent_similarities[i])) for i in range(len(adjacent_similarities)) if abs(adjacent_similarities[i]) > 0.5]
low_sim_pairs = [(i, abs(adjacent_similarities[i])) for i in range(len(adjacent_similarities)) if abs(adjacent_similarities[i]) < 0.2]

if high_sim_pairs:
    print(f"\n高相似度区域（相邻层差异小）：")
    for i, sim in high_sim_pairs:
        print(f"  Layer {i} -> {i+1}: 相似度={sim:.4f}")
        print(f"    → 这几层可能在做类似的事情")
        print(f"    → 可能是模型冗余，可以考虑压缩")

if low_sim_pairs:
    print(f"\n低相似度区域（相邻层差异大）：")
    for i, sim in low_sim_pairs:
        print(f"  Layer {i} -> {i+1}: 相似度={sim:.4f}")
        print(f"    → 这里发生了功能转换")
        print(f"    → 重要的分界线")

# ============================================
# 6. 生成可视化
# ============================================
print("\n【步骤6】生成可视化热力图")
print("-" * 80)

# 准备数据：只分析Transformer的注意力层
attn_layers = [name for _, name, _ in transformer_layers]
n_layers = len(attn_layers)

# 创建相似度矩阵
sim_matrix = np.zeros((n_layers, n_layers))

for i in range(n_layers):
    for j in range(n_layers):
        if i == j:
            sim_matrix[i, j] = 1.0
        elif i < j:
            key = (attn_layers[i], attn_layers[j])
            if key in cosine_sim:
                sim_matrix[i, j] = cosine_sim[key]
            else:
                sim_matrix[i, j] = 0
        else:
            sim_matrix[i, j] = sim_matrix[j, i]

# 绘制热力图
plt.figure(figsize=(14, 12))

# 简化层名显示
layer_labels = [f"Layer {i}" for i in range(n_layers)]

sns.heatmap(sim_matrix,
            xticklabels=layer_labels,
            yticklabels=layer_labels,
            annot=False,
            fmt='.2f',
            cmap='RdYlBu_r',
            vmin=-1,
            vmax=1,
            cbar_kws={'label': '余弦相似度'})

plt.title('GPT-2 Transformer层相似度矩阵（注意力权重）', fontsize=16, pad=20)
plt.xlabel('层编号', fontsize=12)
plt.ylabel('层编号', fontsize=12)
plt.tight_layout()

# 保存图片
output_path = 'outputs/figures/gpt2_layer_similarity.png'
import os
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ 热力图已保存: {output_path}")

plt.close()

# ============================================
# 7. 深入分析：MLP层 vs 注意力层
# ============================================
print("\n【步骤7】MLP层 vs 注意力层对比")
print("-" * 80)

# 收集MLP层和注意力层
attn_weights = []
mlp_weights = []

for name, param in model.named_parameters():
    if 'weight' in name:
        if 'attn.c_attn' in name:
            attn_weights.append((name, param))
        elif 'mlp.c_fc' in name:
            mlp_weights.append((name, param))

print(f"注意力层数量: {len(attn_weights)}")
print(f"MLP层数量: {len(mlp_weights)}")

# 计算注意力层内部的平均相似度
attn_similarities = []
for i in range(len(attn_weights) - 1):
    w1 = attn_weights[i][1].data.flatten().float()[:10000]  # 采样以提高速度
    w2 = attn_weights[i+1][1].data.flatten().float()[:10000]
    sim = torch.nn.functional.cosine_similarity(w1, w2, dim=0).item()
    attn_similarities.append(abs(sim))

# 计算MLP层内部的平均相似度
mlp_similarities = []
for i in range(len(mlp_weights) - 1):
    w1 = mlp_weights[i][1].data.flatten().float()[:10000]
    w2 = mlp_weights[i+1][1].data.flatten().float()[:10000]
    sim = torch.nn.functional.cosine_similarity(w1, w2, dim=0).item()
    mlp_similarities.append(abs(sim))

print(f"\n相邻层的平均相似度：")
print(f"  注意力层: {np.mean(attn_similarities):.4f} ± {np.std(attn_similarities):.4f}")
print(f"  MLP层:     {np.mean(mlp_similarities):.4f} ± {np.std(mlp_similarities):.4f}")

if np.mean(attn_similarities) > np.mean(mlp_similarities):
    print(f"\n  → 注意力层变化较慢（更稳定）")
    print(f"  → MLP层变化较快（更多样）")
else:
    print(f"\n  → MLP层变化较慢（更稳定）")
    print(f"  → 注意力层变化较快（更多样）")

# ============================================
# 8. 总结和建议
# ============================================
print("\n" + "=" * 80)
print("分析总结")
print("=" * 80)

print(f"""
模型: {config.name_or_path}
分析层数: {len(transformer_layers)} 个Transformer层

主要发现：

1️⃣  整体相似度分布
    • 平均相邻层相似度: {mean_sim:.4f}
    • 标准差: {std_sim:.4f}
    • {'相似度较高，层间差异较小' if mean_sim > 0.5 else '相似度适中，各层有独特性'}

2️⃣  层的变化模式
    • 注意力层平均相似度: {np.mean(attn_similarities):.4f}
    • MLP层平均相似度: {np.mean(mlp_similarities):.4f}
    • {'注意力层更稳定' if np.mean(attn_similarities) > np.mean(mlp_similarities) else 'MLP层更稳定'}

3️⃣  压缩潜力
    • {'存在高相似度区域，可以压缩' if len(high_sim_pairs) > 0 else '没有明显冗余，压缩空间有限'}
    • 建议重点关注: {f'Layer {high_sim_pairs[0][0]}-{high_sim_pairs[0][0]+1}' if high_sim_pairs else 'N/A'}

4️⃣  关键分界
    • {'存在功能转换点' if len(low_sim_pairs) > 0 else '层间过渡平滑'}
    • 建议重点关注: {f'Layer {low_sim_pairs[0][0]}-{low_sim_pairs[0][0]+1}' if low_sim_pairs else 'N/A'}

实际应用：

【模型优化】
  • 如果发现高相似度区域 → 可以考虑删除层
  • 如果发现低相似度区域 → 这是关键转换点，保留

【架构设计】
  • 相似度逐渐降低 → 设计合理，每层都在学习新特征
  • 相似度波动大 → 可能需要调整层数

【知识蒸馏】
  • 选择相似度适中的层对进行蒸馏
  • 避免过于相似或差异过大的层对

【迁移学习】
  • 高相似度层可以冻结
  • 低相似度层需要重新训练
""")

print("\n分析完成！✓")
print(f"可视化图表已保存到: {output_path}")

#!/usr/bin/env python3
"""
层相似度分析演示

展示如何使用相似度分析理解模型内部的层关系
"""

import sys
sys.path.insert(0, "/mnt/hdsd1/guomin/projects/model_probe")

import torch
import numpy as np
from model_probe import ModelWrapper, ModelConfig
from model_probe.analysis import WeightAnalyzer

print("=" * 80)
print("层相似度分析演示")
print("=" * 80)

# ============================================
# 创建一个简单的模型用于演示
# ============================================
print("\n【步骤1】创建示例模型")
print("-" * 80)

import torch.nn as nn

# 创建一个有3个隐藏层的模型
model = nn.Sequential(
    nn.Linear(100, 200),
    nn.ReLU(),
    nn.Linear(200, 150),
    nn.ReLU(),
    nn.Linear(150, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)

print("模型结构：")
for i, layer in enumerate(model):
    if isinstance(layer, nn.Linear):
        print(f"  layer_{i}: {layer.in_features} -> {layer.out_features}")

# ============================================
# 创建分析器
# ============================================
print("\n【步骤2】创建相似度分析器")
print("-" * 80)

analyzer = WeightAnalyzer(model)
print("✓ 分析器创建完成")

# ============================================
# 方法1: 余弦相似度分析
# ============================================
print("\n【方法1】余弦相似度分析")
print("-" * 80)

cosine_sim = analyzer.compute_layer_similarity(metric="cosine")

print(f"\n计算了 {len(cosine_sim)} 对层的相似度")
print(f"\n余弦相似度结果（部分）：")
print(f"{'层1':<15} {'层2':<15} {'相似度':>10} {'关系':>20}")
print("-" * 70)

for (layer1, layer2), sim in list(cosine_sim.items())[:5]:
    if sim > 0.8:
        relationship = "非常相似"
    elif sim > 0.5:
        relationship = "较为相似"
    elif sim > 0.2:
        relationship = "中等相似"
    elif sim > 0:
        relationship = "略有相似"
    else:
        relationship = "几乎无关"

    # 简化层名
    l1_short = layer1.split('.')[-1][:12]
    l2_short = layer2.split('.')[-1][:12]

    print(f"{l1_short:<15} {l2_short:<15} {sim:>10.4f} {relationship:>20}")

# ============================================
# 方法2: CKA相似度分析
# ============================================
print("\n【方法2】CKA相似度分析")
print("-" * 80)

print("正在计算CKA相似度（可能需要一些时间）...")

try:
    cka_sim = analyzer.compute_layer_similarity(metric="cka")

    print(f"\n✓ CKA计算完成")
    print(f"计算了 {len(cka_sim)} 对层的相似度")
    print(f"\nCKA相似度结果（部分）：")
    print(f"{'层1':<15} {'层2':<15} {'CKA值':>10} {'关系':>20}")
    print("-" * 70)

    for (layer1, layer2), sim in list(cka_sim.items())[:5]:
        if sim > 0.8:
            relationship = "表示高度相似"
        elif sim > 0.5:
            relationship = "表示较为相似"
        elif sim > 0.3:
            relationship = "表示中等相似"
        else:
            relationship = "表示差异较大"

        l1_short = layer1.split('.')[-1][:12]
        l2_short = layer2.split('.')[-1][:12]

        print(f"{l1_short:<15} {l2_short:<15} {sim:>10.4f} {relationship:>20}")

except Exception as e:
    print(f"CKA计算出错: {e}")

# ============================================
# 方法对比
# ============================================
print("\n【方法对比】余弦相似度 vs CKA")
print("-" * 80)

print("""
┌─────────────┬──────────────┬──────────────┬──────────────┐
│   特性      │  余弦相似度   │     CKA      │              │
├─────────────┼──────────────┼──────────────┼──────────────┤
│ 计算速度    │   ⚡⚡⚡ 快   │  🐌 慢       │              │
│ 不变性      │   对缩放不变  │对旋转/缩放不变│              │
│ 适用场景    │  快速筛选    │  深入分析    │              │
│ 学术认可    │   一般       │  ⭐⭐⭐ 高    │              │
│ 理解难度    │   简单       │  较复杂      │              │
└─────────────┴──────────────┴──────────────┴──────────────┘

建议使用：
1. 初步探索 → 使用余弦相似度（快速）
2. 深入分析 → 使用CKA（更准确）
3. 对比不同模型 → 使用CKA（更公平）
""")

# ============================================
# 实际应用场景
# ============================================
print("\n【实际应用】相似度分析的用途")
print("-" * 80)

applications = [
    {
        "场景": "模型压缩",
        "说明": "发现相似层，可以合并或删除",
        "方法": "如果 layer_i 和 layer_j 相似度 > 0.9，考虑删除一层"
    },
    {
        "场景": "架构搜索",
        "说明": "找出最优的网络深度",
        "方法": "观察相似度随深度的变化，找到最优层数"
    },
    {
        "场景": "知识蒸馏",
        "说明": "选择最适合蒸馏的层",
        "方法": "找到教师模型和学生模型中相似度最高的层对"
    },
    {
        "场景": "迁移学习",
        "说明": "选择要冻结的层",
        "方法": "相似度低的层更值得重新训练"
    },
    {
        "场景": "模型解释",
        "说明": "理解各层学到了什么",
        "方法": "相似度高的层学到了相似的特征"
    }
]

for i, app in enumerate(applications, 1):
    print(f"\n{i}. {app['场景']}")
    print(f"   说明: {app['说明']}")
    print(f"   方法: {app['方法']}")

# ============================================
# 可视化示例
# ============================================
print("\n【可视化】生成相似度热力图")
print("-" * 80)

print("""
相似度热力图示例：

        layer1  layer2  layer3  output
layer1   1.00    0.23    0.15    0.08
layer2   0.23    1.00    0.67    0.34
layer3   0.15    0.67    1.00    0.45
output   0.08    0.34    0.45    1.00

颜色编码：
  🔴 红色 = 高相似度（0.8-1.0）
  🟡 黄色 = 中等相似度（0.4-0.8）
  🔵 蓝色 = 低相似度（0.0-0.4）

从上图可以看出：
  • layer2 和 layer3 相似度较高（0.67）
  • 可能可以考虑合并这两层
  • output层与其他层差异较大
""")

# ============================================
# 代码示例
# ============================================
print("\n【使用示例】代码")
print("-" * 80)

print("""
from model_probe.analysis import WeightAnalyzer

# 1. 创建分析器
analyzer = WeightAnalyzer(model)

# 2. 计算余弦相似度
cosine_sim = analyzer.compute_layer_similarity(metric="cosine")

# 3. 计算CKA相似度
cka_sim = analyzer.compute_layer_similarity(metric="cka")

# 4. 生成可视化热力图
analyzer.visualize_similarity_matrix(
    metric="cosine",
    save_path="outputs/figures/similarity_heatmap.png"
)

# 5. 找出最相似的层对
sorted_sim = sorted(cosine_sim.items(), key=lambda x: x[1], reverse=True)
most_similar = sorted_sim[0]
print(f"最相似的层: {most_similar[0]}")
print(f"相似度: {most_similar[1]:.4f}")
""")

print("\n" + "=" * 80)
print("演示完成！")
print("=" * 80)

print("""
总结：
  ✓ 余弦相似度：快速、简单，适合初步分析
  ✓ CKA相似度：严格、准确，适合深入分析
  ✓ 可视化：生成热力图，直观展示层关系
  ✓ 应用广泛：模型压缩、架构搜索、迁移学习等

建议：
  1. 先用余弦相似度快速扫描
  2. 对重点层使用CKA深入分析
  3. 结合可视化理解整体结构
  4. 根据分析结果优化模型
""")

#!/usr/bin/env python3
"""
静态分析完整演示

展示如何使用静态分析理解模型，无需运行推理
"""

import sys
sys.path.insert(0, "/mnt/hdsd1/guomin/projects/model_probe")

import torch
import torch.nn as nn
import numpy as np
from model_probe.analysis import WeightAnalyzer

print("=" * 70)
print("静态分析完整演示")
print("=" * 70)

# ============================================
# 1. 创建示例模型
# ============================================
print("\n【步骤1】创建示例模型")
print("-" * 70)

# 创建一个简单的多层神经网络
model = nn.Sequential(
    nn.Linear(100, 200),  # 第1层：100 -> 200
    nn.ReLU(),
    nn.Linear(200, 100),  # 第2层：200 -> 100
    nn.ReLU(),
    nn.Linear(100, 10),   # 第3层：100 -> 10
)

print("模型结构：")
print(model)
print(f"\n总参数量: {sum(p.numel() for p in model.parameters()):,}")

# ============================================
# 2. 基础权重统计
# ============================================
print("\n\n【步骤2】基础权重统计")
print("-" * 70)

analyzer = WeightAnalyzer(model)
stats = analyzer.compute_weight_statistics()

print("\n各层权重统计：")
print(f"{'层名':<20} {'均值':>10} {'标准差':>10} {'最小值':>10} {'最大值':>10}")
print("-" * 70)

for name, s in stats.items():
    print(f"{name:<20} {s['mean']:>10.6f} {s['std']:>10.6f} {s['min']:>10.6f} {s['max']:>10.6f}")

# ============================================
# 3. 参数量分布
# ============================================
print("\n\n【步骤3】参数量分布")
print("-" * 70)

param_counts = analyzer.compute_parameter_count()

print("\n各层参数量：")
print(f"{'层名':<30} {'参数量':>15} {'占比':>10}")
print("-" * 70)

total = param_counts['_total']
for name, count in param_counts.items():
    if name != '_total':
        percentage = (count / total) * 100
        print(f"{name:<30} {count:>15,} {percentage:>9.2f}%")

print(f"\n{'总计':<30} {total:>15,}")

# ============================================
# 4. SVD分析 - 理解层的信息容量
# ============================================
print("\n\n【步骤4】SVD分析 - 理解层的信息容量")
print("-" * 70)

# 分析第一层的SVD
layer_name = "0.weight"  # 第一层权重
singular_values, variance_ratio = analyzer.compute_svd(layer_name, k=10)

print(f"\n'{layer_name}' 的前10个奇异值：")
print(f"{'排名':<6} {'奇异值':>12} {'解释方差':>12} {'累计方差':>12}")
print("-" * 70)

cumsum = 0
for i, (sv, vr) in enumerate(zip(singular_values, variance_ratio)):
    cumsum += vr
    print(f"{i+1:<6} {sv:>12.4f} {vr:>11.2%} {cumsum:>11.2%}")

# ============================================
# 5. 有效秩分析 - 模型压缩潜力
# ============================================
print("\n\n【步骤5】有效秩分析 - 模型压缩潜力")
print("-" * 70)

print("\n各层的有效秩：")
print(f"{'层名':<20} {'理论秩':>10} {'有效秩':>10} {'压缩比':>10}")
print("-" * 70)

for name, param in model.named_parameters():
    if "weight" in name:
        theoretical_rank = min(param.shape)
        effective_rank = analyzer.compute_rank(name)

        compression_ratio = theoretical_rank / effective_rank if effective_rank > 0 else 0

        print(f"{name:<20} {theoretical_rank:>10} {effective_rank:>10} {compression_ratio:>10.2f}x")

# ============================================
# 6. 稀疏性分析 - 剪枝潜力
# ============================================
print("\n\n【步骤6】稀疏性分析 - 剪枝潜力")
print("-" * 70)

print("\n各层的稀疏性：")
print(f"{'层名':<20} {'总参数':>12} {'接近零':>12} {'稀疏度':>10}")
print("-" * 70)

for name, s in stats.items():
    # 获取该层的总参数数
    for param_name, param in model.named_parameters():
        if param_name == name:
            total_params = param.numel()
            near_zero = int(s['sparsity'] * total_params)
            print(f"{name:<20} {total_params:>12,} {near_zero:>12,} {s['sparsity']:>9.2%}")

# ============================================
# 7. 实际应用场景
# ============================================
print("\n\n【步骤7】静态分析的实际应用")
print("-" * 70)

print("""
静态分析可以用于：

1️⃣  模型压缩
    • 通过SVD发现哪些层可以压缩
    • 通过稀疏性分析发现可以剪枝的参数
    • 通过有效秩分析降低维度

2️⃣  模型质量评估
    • 检查权重分布是否异常
    • 发现训练问题（梯度消失/爆炸）
    • 比较不同模型的容量

3️⃣  架构搜索
    • 分析哪些层最重要
    • 找出冗余的层
    • 优化网络结构

4️⃣  安全性检查
    • 检测模型是否被篡改
    • 发现异常的权重模式
    • 验证模型完整性

5️⃣  快速筛选
    • 无需推理即可了解模型特性
    • 快速比较多个模型
    • 零成本的初步分析
""")

# ============================================
# 8. 总结
# ============================================
print("\n" + "=" * 70)
print("静态分析总结")
print("=" * 70)

print("""
静态分析的优点：
  ✅ 快速 - 秒级完成
  ✅ 无需GPU - CPU即可运行
  ✅ 无需数据 - 不依赖输入
  ✅ 可解释 - 直接分析权重

静态分析的局限：
  ⚠️  只看结构，不看行为
  ⚠️  无法评估实际性能
  ⚠️  需要结合动态分析

最佳实践：
  🎯 先做静态分析快速了解模型
  🎯 再做动态分析深入理解行为
  🎯 两者结合获得完整视图
""")

print("\n演示完成！")

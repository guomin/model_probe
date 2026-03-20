#!/usr/bin/env python3
"""
真实模型的静态分析演示

使用 GPT-2 模型展示静态分析的实际效果
"""

import sys
sys.path.insert(0, "/mnt/hdsd1/guomin/projects/model_probe")

import torch
from model_probe import ModelWrapper, ModelConfig
from model_probe.analysis import WeightAnalyzer
import json

print("=" * 80)
print("GPT-2 模型静态分析")
print("=" * 80)

# ============================================
# 1. 加载模型
# ============================================
print("\n【步骤1】加载 GPT-2 模型")
print("-" * 80)

config = ModelConfig(
    name_or_path="gpt2",
    device="cpu"  # 静态分析不需要GPU
)

wrapper = ModelWrapper(config)
wrapper.load_model(for_generation=False)

model = wrapper.model

print(f"✓ 模型加载成功")
print(f"  模型名称: {config.name_or_path}")
print(f"  总参数量: {wrapper.get_parameter_count():,}")
print(f"  层数: {len([m for m in model.modules() if hasattr(m, 'weight')])}")

# ============================================
# 2. 创建静态分析器
# ============================================
print("\n【步骤2】创建静态分析器")
print("-" * 80)

analyzer = WeightAnalyzer(model)
print("✓ 分析器创建完成")

# ============================================
# 3. 基础权重统计
# ============================================
print("\n【步骤3】分析各层权重统计")
print("-" * 80)

stats = analyzer.compute_weight_statistics()

print(f"\n共分析 {len(stats)} 个权重矩阵")
print(f"\n前10层的权重统计：")
print(f"{'层名':<40} {'维度':>20} {'均值':>10} {'标准差':>10} {'稀疏度':>10}")
print("-" * 100)

count = 0
for name, s in stats.items():
    if count >= 10:
        break

    # 获取权重形状
    for param_name, param in model.named_parameters():
        if param_name == name:
            shape = str(list(param.shape))
            break

    print(f"{name:<40} {shape:>20} {s['mean']:>10.6f} {s['std']:>10.6f} {s['sparsity']:>9.2%}")
    count += 1

print(f"\n... (还有 {len(stats) - 10} 层)")

# ============================================
# 4. 参数量分布
# ============================================
print("\n【步骤4】参数量分布分析")
print("-" * 80)

param_counts = analyzer.compute_parameter_count()
total = param_counts['_total']

print(f"\n总参数量: {total:,}")

# 统计不同类型的参数
type_counts = {}
for name, count in param_counts.items():
    if name == '_total':
        continue

    # 提取参数类型
    if 'weight' in name:
        param_type = '权重权重'
    elif 'bias' in name:
        param_type = '偏置'
    else:
        param_type = '其他'

    if param_type not in type_counts:
        type_counts[param_type] = 0
    type_counts[param_type] += count

print(f"\n按类型统计：")
for param_type, count in type_counts.items():
    percentage = (count / total) * 100
    print(f"  {param_type}: {count:,} ({percentage:.2f}%)")

# ============================================
# 5. 各层参数量分布
# ============================================
print("\n【步骤5】各层参数量分布")
print("-" * 80)

layer_params = []
for name, count in param_counts.items():
    if name == '_total':
        continue
    layer_params.append((name, count))

# 按参数量排序，显示前10和后5
layer_params.sort(key=lambda x: x[1], reverse=True)

print(f"\n参数量最多的10层：")
print(f"{'排名':<6} {'层名':<50} {'参数量':>15} {'占比':>10}")
print("-" * 85)

for i, (name, count) in enumerate(layer_params[:10]):
    percentage = (count / total) * 100
    print(f"{i+1:<6} {name:<50} {count:>15,} {percentage:>9.2f}%")

print(f"\n参数量最少的5层：")
print(f"{'排名':<6} {'层名':<50} {'参数量':>15} {'占比':>10}")
print("-" * 85)

for i, (name, count) in enumerate(layer_params[-5:]):
    percentage = (count / total) * 100
    print(f"{i+1:<6} {name:<50} {count:>15,} {percentage:>9.2f}%")

# ============================================
# 6. SVD分析 - 关键层的信息容量
# ============================================
print("\n【步骤6】SVD分析 - 关键层的信息容量")
print("-" * 80)

# 选择几个关键层进行SVD分析
key_layers = [
    'transformer.h.0.attn.c_attn.weight',  # 第1层注意力
    'transformer.h.5.attn.c_attn.weight',  # 中间层注意力
    'transformer.h.11.attn.c_attn.weight', # 最后一层注意力
]

for layer_name in key_layers:
    try:
        singular_values, variance_ratio = analyzer.compute_svd(layer_name, k=10)

        print(f"\n{layer_name}:")
        print(f"  前10个奇异值解释的方差: {variance_ratio.sum():.2%}")
        print(f"  前5个奇异值: {singular_values[:5]}")

        # 计算需要多少个奇异值才能解释80%的方差
        cumsum = 0
        for i, vr in enumerate(variance_ratio):
            cumsum += vr
            if cumsum >= 0.8:
                print(f"  需要前 {i+1} 个奇异值解释80%方差")
                break

    except Exception as e:
        print(f"\n{layer_name}: 分析失败 - {e}")

# ============================================
# 7. 有效秩分析 - 压缩潜力
# ============================================
print("\n【步骤7】有效秩分析 - 压缩潜力")
print("-" * 80)

print(f"\n关键层的有效秩：")
print(f"{'层名':<50} {'理论秩':>10} {'有效秩':>10} {'压缩比':>10} {'潜力':>10}")
print("-" * 100)

for layer_name in key_layers:
    try:
        # 获取理论秩
        for param_name, param in model.named_parameters():
            if param_name == layer_name:
                theoretical_rank = min(param.shape)
                break

        effective_rank = analyzer.compute_rank(layer_name)
        compression_ratio = theoretical_rank / effective_rank if effective_rank > 0 else 0

        # 判断压缩潜力
        if compression_ratio > 2:
            potential = "高"
        elif compression_ratio > 1.5:
            potential = "中"
        else:
            potential = "低"

        print(f"{layer_name:<50} {theoretical_rank:>10} {effective_rank:>10} {compression_ratio:>10.2f}x {potential:>10}")

    except Exception as e:
        print(f"{layer_name:<50} 分析失败: {e}")

# ============================================
# 8. 发现和洞察
# ============================================
print("\n【步骤8】分析洞察")
print("-" * 80)

insights = []

# 洞察1: 参数分布
if len(layer_params) > 0:
    max_param_layer = layer_params[0]
    min_param_layer = layer_params[-1]
    insights.append(f"• 参数量最大的层是 {max_param_layer[0]}，有 {max_param_layer[1]:,} 个参数")
    insights.append(f"• 参数量最小的层是 {min_param_layer[0]}，有 {min_param_layer[1]:,} 个参数")

# 洞察2: 稀疏性
total_sparse = sum(s['sparsity'] for s in stats.values())
avg_sparsity = total_sparse / len(stats)
insights.append(f"• 平均稀疏度为 {avg_sparsity:.2%}")

if avg_sparsity > 0.1:
    insights.append(f"  → 模型较稀疏，有剪枝潜力")
else:
    insights.append(f"  → 模型较稠密，剪枝空间有限")

# 洞察3: 权重分布
all_means = [s['mean'] for s in stats.values()]
all_stds = [s['std'] for s in stats.values()]
insights.append(f"• 权重均值范围: {min(all_means):.6f} ~ {max(all_means):.6f}")
insights.append(f"• 权重标准差范围: {min(all_stds):.6f} ~ {max(all_stds):.6f}")

if max(all_stds) / min(all_stds) > 10:
    insights.append(f"  → 不同层的权重分布差异很大")

print("\n关键发现：")
for insight in insights:
    print(insight)

# ============================================
# 9. 保存分析结果
# ============================================
print("\n【步骤9】保存分析结果")
print("-" * 80)

results = {
    'model_name': config.name_or_path,
    'total_parameters': total,
    'weight_statistics': stats,
    'parameter_counts': param_counts,
    'insights': insights
}

output_file = 'outputs/reports/static_analysis_report.json'
import os
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✓ 分析结果已保存到: {output_file}")

# ============================================
# 10. 总结
# ============================================
print("\n" + "=" * 80)
print("静态分析完成")
print("=" * 80)

print(f"""
分析总结：
  ✓ 模型: {config.name_or_path}
  ✓ 总参数量: {total:,}
  ✓ 分析层数: {len(stats)}
  ✓ 完成时间: {torch.utils.data.get_worker_info() if hasattr(torch.utils.data, 'get_worker_info') else 'N/A'}

静态分析的优点：
  ⚡ 快速 - 无需运行推理
  💰 零成本 - 不需要GPU
  🔍 深入 - 了解模型内部结构
  📊 可视化 - 生成详细报告

下一步建议：
  1. 根据SVD分析结果进行模型压缩
  2. 根据稀疏性分析进行剪枝
  3. 对比不同模型的静态特性
  4. 结合动态分析获得完整视图
""")

print("\n分析完成！✓")

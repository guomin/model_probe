#!/usr/bin/env python3
"""
动态层相似性分析 - 基于激活值

运行模型推理，捕获中间层的激活值，然后分析层与层之间的相似性
"""

import sys
sys.path.insert(0, "/mnt/hdsd1/guomin/projects/model_probe")

import torch
import numpy as np
from model_probe import ModelWrapper, ModelConfig, HookManager
from model_probe.analysis import ActivationAnalyzer
import json
from datetime import datetime

print("=" * 80)
print("动态层相似性分析 - 基于激活值")
print("=" * 80)

# ============================================
# 1. 加载模型
# ============================================
print("\n【步骤1】加载模型")
print("-" * 80)

config = ModelConfig(
    name_or_path="gpt2",
    device="cpu"
)

wrapper = ModelWrapper(config)
wrapper.load_model(for_generation=False)

print(f"✓ 模型加载成功: {config.name_or_path}")
print(f"  总参数量: {wrapper.get_parameter_count():,}")

# ============================================
# 2. 准备测试数据
# ============================================
print("\n【步骤2】准备测试数据")
print("-" * 80)

# 准备一些测试句子
test_sentences = [
    "The cat sits on the mat.",
    "Artificial intelligence is transforming the world.",
    "Python is a popular programming language.",
    "Machine learning models can understand patterns.",
    "The quick brown fox jumps over the lazy dog.",
]

print(f"准备了 {len(test_sentences)} 个测试句子")

# Tokenize所有句子
all_input_ids = []
max_length = 0

for sentence in test_sentences:
    inputs = wrapper.tokenizer(sentence, return_tensors="pt")
    input_ids = inputs["input_ids"]
    all_input_ids.append(input_ids)
    max_length = max(max_length, input_ids.shape[1])

# Padding到相同长度
padded_inputs = []
for input_ids in all_input_ids:
    if input_ids.shape[1] < max_length:
        pad_length = max_length - input_ids.shape[1]
        padding = torch.zeros((1, pad_length), dtype=torch.long)
        input_ids = torch.cat([input_ids, padding], dim=1)
    padded_inputs.append(input_ids)

# 合并成一个batch
batch_input_ids = torch.cat(padded_inputs, dim=0)

print(f"  Batch大小: {batch_input_ids.shape[0]}")
print(f"  序列长度: {batch_input_ids.shape[1]}")

# ============================================
# 3. 捕获激活值
# ============================================
print("\n【步骤3】捕获中间层激活值")
print("-" * 80)

# 使用HookManager捕获激活值
hook_manager = HookManager(wrapper.model)

# 注册所有Transformer层的Hook
layer_names = []
for name, module in wrapper.model.named_modules():
    if 'h.' in name and 'attn.c_attn' in name:
        layer_names.append(name)

print(f"注册了 {len(layer_names)} 个层的Hook")

hook_manager.register_hooks(layer_names, hook_type="forward")

# 运行推理
print("正在运行模型推理...")
with torch.no_grad():
    outputs = wrapper.model(
        input_ids=batch_input_ids,
        output_hidden_states=True,
        use_cache=False
    )

# 获取激活值
activations = hook_manager.get_activations(clear=False)

print(f"✓ 捕获了 {len(activations)} 层的激活值")

for name, acts in activations.items():
    print(f"  {name}: shape = {acts[0].shape}")

# ============================================
# 4. 计算激活相似度（使用简化的CKA）
# ============================================
print("\n【步骤4】计算激活相似度")
print("-" * 80)

def compute_linear_cka(activations1, activations2):
    """
    计算线性CKA (Linear Centered Kernel Alignment)

    这是一个简化版本，更高效
    """
    # 处理list格式的激活值
    if isinstance(activations1, list):
        acts1 = activations1[0]  # 取第一个batch
    else:
        acts1 = activations1

    if isinstance(activations2, list):
        acts2 = activations2[0]
    else:
        acts2 = activations2

    # 展平: (seq, hidden) -> (seq * hidden,)
    acts1_flat = acts1.flatten()
    acts2_flat = acts2.flatten()

    # 计算余弦相似度
    similarity = torch.nn.functional.cosine_similarity(
        acts1_flat.unsqueeze(0),
        acts2_flat.unsqueeze(0),
        dim=1
    ).item()

    return similarity

def compute_representation_similarity(activations1, activations2):
    """
    计算表示相似度 - 基于激活值的中心化核对齐
    """
    # 处理list格式的激活值
    if isinstance(activations1, list):
        acts1 = torch.cat(activations1, dim=0)  # (total_tokens, hidden_dim)
    else:
        acts1 = activations1

    if isinstance(activations2, list):
        acts2 = torch.cat(activations2, dim=0)
    else:
        acts2 = activations2

    # 中心化
    acts1_centered = acts1 - acts1.mean(dim=0, keepdim=True)
    acts2_centered = acts2 - acts2.mean(dim=0, keepdim=True)

    # 计算相似度
    # 使用Frobenius内积
    similarity = (acts1_centered * acts2_centered).sum()
    norm1 = (acts1_centered * acts1_centered).sum().sqrt()
    norm2 = (acts2_centered * acts2_centered).sum().sqrt()

    if norm1 > 0 and norm2 > 0:
        similarity = similarity / (norm1 * norm2)
    else:
        similarity = 0.0

    return similarity.item()

# 计算所有层对之间的相似度
print("正在计算激活相似度...")

layer_similarities = {}
layer_names_sorted = sorted(activations.keys())

for i, name1 in enumerate(layer_names_sorted):
    for j, name2 in enumerate(layer_names_sorted):
        if i < j:
            # 计算两种相似度
            sim_linear = compute_linear_cka(activations[name1], activations[name2])
            sim_repr = compute_representation_similarity(activations[name1], activations[name2])

            layer_similarities[(name1, name2)] = {
                'linear_cka': sim_linear,
                'representation_similarity': sim_repr
            }

print(f"✓ 计算了 {len(layer_similarities)} 对层的激活相似度")

# ============================================
# 5. 分析结果
# ============================================
print("\n【步骤5】分析结果")
print("-" * 80)

# 提取表示相似度
repr_similarities = [(pair, sim['representation_similarity']) for pair, sim in layer_similarities.items()]
sorted_repr = sorted(repr_similarities, key=lambda x: abs(x[1]), reverse=True)

print(f"\n表示相似度最高的10对层（基于激活值）：")
print(f"{'排名':<6} {'层1':<40} {'层2':<40} {'相似度':>10}")
print("-" * 110)

for i, ((layer1, layer2), sim) in enumerate(sorted_repr[:10], 1):
    # 提取层号
    l1_num = layer1.split('.')[1]
    l2_num = layer2.split('.')[1]

    print(f"{i:<6} {layer1:<40} {layer2:<40} {abs(sim):>10.4f}")

# 统计分析
all_sims = [abs(sim['representation_similarity']) for _, sim in layer_similarities.items()]

print(f"\n统计分析：")
print(f"  平均相似度: {np.mean(all_sims):.4f}")
print(f"  标准差: {np.std(all_sims):.4f}")
print(f"  最大相似度: {np.max(all_sims):.4f}")
print(f"  最小相似度: {np.min(all_sims):.4f}")

# 相邻层分析
print(f"\n相邻层的激活相似度：")
print(f"{'层转换':<20} {'表示相似度':>15} {'趋势':>20}")
print("-" * 60)

adjacent_acts = []
for i in range(len(layer_names_sorted) - 1):
    layer1 = layer_names_sorted[i]
    layer2 = layer_names_sorted[i + 1]

    key = (layer1, layer2)
    if key in layer_similarities:
        sim = layer_similarities[key]['representation_similarity']
    else:
        key = (layer2, layer1)
        sim = layer_similarities[key]['representation_similarity']

    adjacent_acts.append(abs(sim))

    l1_num = layer1.split('.')[1]
    l2_num = layer2.split('.')[1]

    if abs(sim) > 0.8:
        trend = "高度相似"
    elif abs(sim) > 0.5:
        trend = "较为相似"
    elif abs(sim) > 0.3:
        trend = "中等相似"
    else:
        trend = "差异较大"

    print(f"Layer {l1_num} → {l2_num}  {abs(sim):>14.4f}      {trend:>20}")

# ============================================
# 6. 对比静态vs动态
# ============================================
print("\n【步骤6】对比：静态权重 vs 动态激活")
print("-" * 80)

print("""
静态分析（权重）：
  • 分析模型的参数
  • 不需要运行推理
  • 反映模型的"能力"

动态分析（激活值）：
  • 分析模型的实际输出
  • 需要运行推理
  • 反映模型的"行为"

关键区别：
  • 静态相似度高 → 权重结构相似
  • 动态相似度高 → 实际行为相似

通常情况：
  • 静态相似度低，动态相似度也低 → 每层都很独特
  • 静态相似度高，动态相似度低 → 权重相似但功能不同
  • 静态相似度低，动态相似度高 → 权重不同但功能相似
""")

# ============================================
# 7. 保存结果
# ============================================
print("\n【步骤7】保存结果")
print("-" * 80)

results = {
    'model_name': config.name_or_path,
    'analysis_type': 'dynamic_activation_similarity',
    'num_layers': len(layer_names_sorted),
    'num_sentences': len(test_sentences),
    'statistics': {
        'mean_similarity': float(np.mean(all_sims)),
        'std_similarity': float(np.std(all_sims)),
        'max_similarity': float(np.max(all_sims)),
        'min_similarity': float(np.min(all_sims))
    },
    'top_similar': [
        {
            'layer1': layer1,
            'layer2': layer2,
            'representation_similarity': float(sim)
        }
        for (layer1, layer2), sim in sorted_repr[:10]
    ],
    'adjacent_similarities': [
        {
            'from': int(layer_names_sorted[i].split('.')[1]),
            'to': int(layer_names_sorted[i+1].split('.')[1]),
            'similarity': float(adjacent_acts[i])
        }
        for i in range(len(adjacent_acts))
    ]
}

output_path = 'outputs/reports/dynamic_similarity_analysis.json'
import os
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✓ 结果已保存到: {output_path}")

# ============================================
# 8. 总结
# ============================================
print("\n" + "=" * 80)
print("动态相似性分析完成")
print("=" * 80)

print(f"""
模型: {config.name_or_path}
分析方法: 基于激活值的动态分析
测试句子: {len(test_sentences)} 个
分析层数: {len(layer_names_sorted)}

主要发现：

1️⃣  激活相似度分布
    • 平均相似度: {np.mean(all_sims):.4f}
    • {'各层激活较为相似，可能有冗余' if np.mean(all_sims) > 0.5 else '各层激活差异较大，功能分工明确'}

2️⃣  相邻层变化
    • 平均相邻相似度: {np.mean(adjacent_acts):.4f}
    • {'变化较慢，表示逐步演化' if np.mean(adjacent_acts) > 0.5 else '变化较快，表示功能转换'}

3️⃣  与静态分析对比
    • 静态分析看权重结构
    • 动态分析看实际行为
    • 两者结合才能全面理解模型

实际应用：

【模型优化】
  • 动态相似度高 → 可以考虑合并或删除
  • 动态相似度低 → 每层都很重要

【知识蒸馏】
  • 选择动态相似度适中的层对
  • 确保学生模型学到真实的行为

【模型解释】
  • 理解哪些层在做相似的事情
  • 发现模型的信息处理流程
""")

print("\n分析完成！✓")

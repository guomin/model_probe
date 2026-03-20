#!/usr/bin/env python3
"""
生成交互式层相似度分析报告
"""

import sys
sys.path.insert(0, "/mnt/hdsd1/guomin/projects/model_probe")

import torch
import numpy as np
from model_probe import ModelWrapper, ModelConfig
from model_probe.analysis import WeightAnalyzer
from model_probe.reporting import ReportGenerator
import json
from datetime import datetime

print("=" * 80)
print("生成交互式层相似度分析报告")
print("=" * 80)

# ============================================
# 1. 加载模型并计算相似度
# ============================================
print("\n【步骤1】加载模型并计算相似度...")
print("-" * 80)

config = ModelConfig(name_or_path="gpt2", device="cpu")
wrapper = ModelWrapper(config)
wrapper.load_model(for_generation=False)

analyzer = WeightAnalyzer(wrapper.model)

print("正在计算余弦相似度...")
cosine_sim = analyzer.compute_layer_similarity(metric="cosine")
print(f"✓ 计算完成！分析了 {len(cosine_sim)} 对层")

# ============================================
# 2. 准备报告数据
# ============================================
print("\n【步骤2】准备报告数据...")
print("-" * 80)

# 找出最相似的10对层
sorted_sim = sorted(cosine_sim.items(), key=lambda x: abs(x[1]), reverse=True)
top_similar = []
for (layer1, layer2), sim in sorted_sim[:10]:
    # 确定颜色
    if abs(sim) > 0.9:
        color = "#d73027"  # 红色
    elif abs(sim) > 0.5:
        color = "#fee08b"  # 黄色
    else:
        color = "#4575b4"  # 蓝色

    top_similar.append({
        'layer1': layer1,
        'layer2': layer2,
        'similarity': abs(sim),
        'color': color
    })

# 提取相邻Transformer层的相似度
adjacent_layers = []
attn_weights = []
for name, param in wrapper.model.named_parameters():
    if 'h.' in name and 'attn.c_attn.weight' in name:
        layer_num = int(name.split('.')[1])
        attn_weights.append((layer_num, name, param))

attn_weights.sort()

for i in range(len(attn_weights) - 1):
    layer1_num, layer1_name, layer1_weight = attn_weights[i]
    layer2_num, layer2_name, layer2_weight = attn_weights[i + 1]

    w1 = layer1_weight.data.flatten().float()
    w2 = layer2_weight.data.flatten().float()
    min_len = min(len(w1), len(w2))
    sim = torch.nn.functional.cosine_similarity(w1[:min_len], w2[:min_len], dim=0).item()

    # 解释
    if abs(sim) > 0.5:
        interpretation = "较为相似，可能有些冗余"
        color = "#fee08b"
    elif abs(sim) > 0.2:
        interpretation = "中等相似，各有特色"
        color = "#ffffbf"
    else:
        interpretation = "差异较大，都很重要"
        color = "#4575b4"

    adjacent_layers.append({
        'from': layer1_num,
        'to': layer2_num,
        'similarity': abs(sim),
        'interpretation': interpretation,
        'color': color
    })

# 统计数据
similarities = [abs(s) for _, s in cosine_sim.items() if 'attn.c_attn' in _[0] and 'attn.c_attn' in _[1]]
stats = {
    'mean': np.mean(similarities) if similarities else 0,
    'std': np.std(similarities) if similarities else 0,
    'max': np.max(similarities) if similarities else 0,
    'min': np.min(similarities) if similarities else 0
}

# MLP vs 注意力
attn_similarities = []
mlp_similarities = []
attn_weights_all = []
mlp_weights = []

for name, param in wrapper.model.named_parameters():
    if 'weight' in name:
        if 'attn.c_attn' in name:
            attn_weights_all.append((name, param))
        elif 'mlp.c_fc' in name:
            mlp_weights.append((name, param))

for i in range(len(attn_weights_all) - 1):
    w1 = attn_weights_all[i][1].data.flatten().float()[:10000]
    w2 = attn_weights_all[i+1][1].data.flatten().float()[:10000]
    sim = torch.nn.functional.cosine_similarity(w1, w2, dim=0).item()
    attn_similarities.append(abs(sim))

for i in range(len(mlp_weights) - 1):
    w1 = mlp_weights[i][1].data.flatten().float()[:10000]
    w2 = mlp_weights[i+1][1].data.flatten().float()[:10000]
    sim = torch.nn.functional.cosine_similarity(w1, w2, dim=0).item()
    mlp_similarities.append(abs(sim))

mlp_mean = np.mean(mlp_similarities) if mlp_similarities else 0
attn_mean = np.mean(attn_similarities) if attn_similarities else 0

# 准备模板数据
report_data = {
    'model_name': 'GPT-2',
    'num_layers': 12,
    'num_pairs': len(cosine_sim),
    'mean_similarity': stats['mean'],
    'mlp_mean': mlp_mean,
    'attn_mean': attn_mean,
    'top_similar': top_similar,
    'adjacent_layers': adjacent_layers,
    'stats': stats,
    'heatmap_path': 'gpt2_layer_similarity.png'  # 相对路径
}

# ============================================
# 3. 生成HTML报告
# ============================================
print("\n【步骤3】生成HTML报告...")
print("-" * 80)

from jinja2 import Environment, FileSystemLoader

template_dir = "/mnt/hdsd1/guomin/projects/model_probe/model_probe/reporting/templates"
env = Environment(loader=FileSystemLoader(template_dir))
env.filters['formatPercent'] = lambda x: f"{x * 100:.4f}%"

template = env.get_template("similarity.html")

html = template.render(
    title="GPT-2 层相似度分析报告",
    subtitle="理解模型内部的层关系",
    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    data=json.dumps(report_data, ensure_ascii=False, indent=2),
    **report_data
)

# 保存报告
output_path = "outputs/interactive_report/similarity.html"
import os
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"✓ HTML报告已生成: {output_path}")

# ============================================
# 4. 复制热力图
# ============================================
print("\n【步骤4】复制热力图...")
print("-" * 80)

import shutil
heatmap_src = "outputs/figures/gpt2_layer_similarity.png"
heatmap_dst = "outputs/interactive_report/gpt2_layer_similarity.png"

if os.path.exists(heatmap_src):
    shutil.copy(heatmap_src, heatmap_dst)
    print(f"✓ 热力图已复制")
else:
    print(f"⚠️  热力图不存在，请先运行 gpt2_similarity_analysis.py")

# ============================================
# 5. 完成
# ============================================
print("\n" + "=" * 80)
print("报告生成完成！")
print("=" * 80)

print(f"""
✓ HTML报告: {output_path}
✓ 热力图: {heatmap_dst}

在浏览器中打开查看：
  file://{os.path.abspath(output_path)}

或访问HTTP服务：
  http://192.168.0.202:8888/similarity.html
""")

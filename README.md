# Model Probe

神经网络模型内部理解、分析、编辑框架

揭开模型黑盒，让 AI 白化与透明化。

## 特性

- **通用性**: 支持多种模型架构 (Transformer, CNN, RNN)
- **功能全面**: 理解 → 掌握 → 利用 → 改造
- **静态+动态**: 权重分析与激活分析结合
- **模块化设计**: 各模块可独立使用
- **显存优化**: 支持16G显存运行7B模型

## 安装

```bash
cd model_probe
pip install -r requirements.txt
```

## 框架结构

```
model_probe/
├── core/               # 核心模块
│   ├── wrapper.py      # 模型封装
│   └── config.py       # 配置管理
├── probes/             # 探针
│   └── linear.py
├── analysis/           # 分析
│   ├── static.py       # 静态分析（权重/参数）
│   └── attributor.py   # 动态分析
├── visualize/          # 可视化
│   └── attention_viz.py
├── editor/             # 编辑
│   └── knowledge_editor.py
├── verify/             # 验证
│   └── evaluator.py
├── examples/           # 示例脚本
├── tests/              # 测试
├── outputs/            # 输出
│   ├── reports/
│   ├── figures/
│   └── logs/
├── docs/               # 文档
└── requirements.txt
```

## 快速开始

```python
from model_probe import ModelWrapper, ModelConfig

config = ModelConfig(name_or_path="gpt2")
wrapper = ModelWrapper(config)
wrapper.load_model()

inputs = wrapper.tokenizer("Hello world", return_tensors="pt")
hidden_states = wrapper.get_hidden_states(inputs["input_ids"])
print(hidden_states["last_hidden_state"].shape)
```

## 核心模块

### 1. 静态分析 (analysis/)

直接分析模型权重，不需要跑推理。

```python
from model_probe.analysis import WeightAnalyzer

analyzer = WeightAnalyzer(model)
stats = analyzer.compute_weight_statistics()
```

### 2. 探针 (probes/)

训练探针检测知识分布。

```python
from model_probe.probes import LinearProbe, ProbeConfig

probe = LinearProbe(ProbeConfig(hidden_dim=768, num_classes=2))
probe.fit(X_train, y_train)
accuracy = probe.score(X_test, y_test)
```

### 3. 分析 (analysis/)

动态分析激活模式。

```python
from model_probe.analysis import ActivationAnalyzer

analyzer = ActivationAnalyzer(wrapper)
attention_patterns = analyzer.analyze_attention_patterns(input_ids)
```

### 4. 可视化 (visualize/)

生成人类可理解的图表。

```python
from model_probe.visualize import AttentionVisualizer

viz = AttentionVisualizer()
viz.plot_attention_head(attention, tokens, "output.png")
```

### 5. 编辑 (editor/)

修改模型知识。

```python
from model_probe.editor import KnowledgeEditor, EditConfig

editor = KnowledgeEditor(wrapper, EditConfig(method="lora", rank=4))
editor.apply_lora()
```

## 运行演示

```bash
# GPT-2 模型分析
python examples/scripts/understand_model.py

# ASR 模型分析 (模拟)
python examples/scripts/asr_simulation.py

# 完整演示
python examples/scripts/demo.py
```

## 显存优化

```python
from model_probe import MemoryOptimizer

MemoryOptimizer.enable_gradient_checkpointing(model)
MemoryOptimizer.enable_cpu_offload(model)
```

## 依赖

- Python >= 3.10
- PyTorch
- Transformers
- scikit-learn
- NumPy, Pandas, Matplotlib, Seaborn

## 参考

- [Linear Probing](https://arxiv.org/abs/2104.14294)
- [Wav2Vec 2.0](https://arxiv.org/abs/2006.11477)
- [ROME](https://arxiv.org/abs/2202.05262)
- [LoRA](https://arxiv.org/abs/2106.09685)

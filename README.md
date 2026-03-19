# Model Probe

神经网络模型内部理解、分析、编辑框架

## 特性

- **通用性**: 支持多种模型架构 (Transformer, CNN, RNN)
- **功能全面**: 理解 → 分析 → 编辑 → 影响
- **显存优化**: 支持16G显存运行7B模型
- **模块化设计**: 核心、探针、分析、编辑分离

## 安装

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install torch transformers sklearn numpy matplotlib pandas tqdm seaborn
```

## 快速开始

```python
from model_probe.core import ModelWrapper, ModelConfig

# 加载模型
config = ModelConfig(name_or_path="gpt2")
wrapper = ModelWrapper(config)
wrapper.load_model()

# 获取隐藏层表示
inputs = wrapper.tokenizer("Hello world", return_tensors="pt")
hidden_states = wrapper.get_hidden_states(inputs["input_ids"])
print(hidden_states["last_hidden_state"].shape)
```

## 框架结构

```
model_probe/
├── core/           # 核心模块
│   ├── ModelWrapper    # 模型封装
│   ├── HookManager     # Hook管理
│   └── MemoryOptimizer # 显存优化
│
├── probes/         # 探针模块
│   ├── LinearProbe     # 线性探针
│   └── RepresentationAnalyzer  # 表示分析
│
├── analysis/       # 分析模块
│   ├── ActivationAnalyzer  # 激活分析
│   └── Attributor         # 归因分析
│
└── editor/        # 编辑模块
    ├── KnowledgeEditor    # 知识编辑
    └── Locator            # 知识定位
```

## 使用示例

### 1. 线性探针 - 探测语法知识

```python
from model_probe.probes import LinearProbe, ProbeConfig

# 训练探针
probe = LinearProbe(ProbeConfig(hidden_dim=768, num_classes=2))
probe.fit(X_train, y_train)

# 评估
accuracy = probe.score(X_test, y_test)
```

### 2. 注意力可视化

```python
from model_probe.analysis import ActivationAnalyzer

analyzer = ActivationAnalyzer(wrapper)
attention_patterns = analyzer.analyze_attention_patterns(input_ids)
analyzer.visualize_attention(input_ids, tokens)
```

### 3. 知识编辑 (LoRA)

```python
from model_probe.editor import KnowledgeEditor, EditConfig

editor = KnowledgeEditor(wrapper, EditConfig(method="lora", rank=4))
editor.apply_lora()
editor.finetune_edit(train_data)
```

## 运行演示

```bash
# GPT-2 模型分析
python understand_model.py

# ASR 模型分析 (模拟)
python asr_simulation.py

# 完整演示
python demo.py
```

## 分析报告

| 模型 | 任务 | 最佳层 | 准确率 |
|------|------|--------|--------|
| GPT-2 | 语法识别 | 全部层 | 90% |
| GPT-2 | 实体类型 | Layer 12 | 94% |
| GPT-2 | 数字推理 | Layer 0 | 61% |
| Wav2Vec (模拟) | 说话人识别 | Layer 5-11 | 100% |
| Wav2Vec (模拟) | 语言识别 | Layer 10 | 90% |
| Wav2Vec (模拟) | 情感识别 | Layer 11 | 100% |

## 显存优化

在16G显存下运行7B模型：

```python
from model_probe.core import MemoryOptimizer

# 启用梯度检查点
MemoryOptimizer.enable_gradient_checkpointing(model)

# 启用CPU卸载
MemoryOptimizer.enable_cpu_offload(model)
```

## 依赖

- Python >= 3.10
- PyTorch
- Transformers
- scikit-learn
- NumPy, Pandas, Matplotlib
- Seaborn

## 许可

MIT License

## 参考

- [Linear Probing](https://arxiv.org/abs/2104.14294)
- [Wav2Vec 2.0](https://arxiv.org/abs/2006.11477)
- [ROME](https://arxiv.org/abs/2202.05262)
- [LoRA](https://arxiv.org/abs/2106.09685)

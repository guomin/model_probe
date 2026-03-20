"""
核心配置管理
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """模型配置类

    Attributes:
        name_or_path: 模型名称或路径
        device: 运行设备
        dtype: 数据类型
        use_cache: 是否使用缓存
        add_special_tokens: 是否添加特殊token
        max_length: 最大长度
        output_attentions: 是否输出注意力权重
    """
    name_or_path: str = "gpt2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    use_cache: bool = False
    add_special_tokens: bool = True
    max_length: int = 512
    output_attentions: bool = True


@dataclass
class ProbeConfig:
    """探针配置类"""
    hidden_dim: int = 768
    num_classes: int = 2
    learning_rate: float = 0.001
    max_epochs: int = 10
    batch_size: int = 32
    test_size: float = 0.2
    random_state: int = 42
    probe_type: str = "linear"  # linear, mlp


@dataclass
class AnalysisConfig:
    """分析配置类"""
    layer_indices: Optional[list] = None  # 要分析的层索引，None表示全部
    head_indices: Optional[list] = None  # 要分析的注意力头索引
    batch_size: int = 8
    output_dir: str = "outputs"
    save_intermediate: bool = True


@dataclass
class EditorConfig:
    """编辑器配置类"""
    method: str = "lora"  # lora, rome, pint
    rank: int = 4
    alpha: float = 32.0
    target_layers: Optional[list] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

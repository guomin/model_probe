"""
Model Probe - 神经网络模型内部理解、分析、编辑框架
揭开模型黑盒，让 AI 白化与透明化
"""

# 核心模块
from .core import (
    ModelWrapper,
    HookManager,
    MemoryOptimizer,
    ModelConfig,
    ProbeConfig,
    AnalysisConfig,
    EditorConfig,
)

# 探针
from .probes import LinearProbe, RepresentationAnalyzer

# 分析
from .analysis import ActivationAnalyzer, Attributor, WeightAnalyzer

# 可视化
from .visualize import AttentionVisualizer, LayerVisualizer, ProbeVisualizer

# 编辑
from .editor import KnowledgeEditor, Locator, EditConfig

# 验证
from .verify import ModelEvaluator, EditResult

__version__ = "0.1.0"

__all__ = [
    # 核心模块
    "ModelWrapper",
    "HookManager",
    "MemoryOptimizer",
    "ModelConfig",
    "ProbeConfig",
    "AnalysisConfig",
    "EditorConfig",
    # 探针
    "LinearProbe",
    "RepresentationAnalyzer",
    # 分析
    "ActivationAnalyzer",
    "Attributor",
    "WeightAnalyzer",
    # 可视化
    "AttentionVisualizer",
    "LayerVisualizer",
    "ProbeVisualizer",
    # 编辑
    "KnowledgeEditor",
    "Locator",
    "EditConfig",
    # 验证
    "ModelEvaluator",
    "EditResult",
]

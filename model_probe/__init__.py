"""
Model Probe - 神经网络模型内部理解、分析、编辑框架
"""

from .core import ModelWrapper, HookManager, MemoryOptimizer
from .probes import LinearProbe, RepresentationAnalyzer
from .analysis import ActivationAnalyzer, Attributor
from .editor import KnowledgeEditor, Locator

__version__ = "0.1.0"

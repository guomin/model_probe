"""
交互式报告生成模块

将模型分析结果转换为人类友好的交互式报告
"""

from .generator import ReportGenerator, QuickReportGenerator

__all__ = [
    "ReportGenerator",
    "QuickReportGenerator",
]

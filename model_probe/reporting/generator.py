"""
交互式报告生成器

将模型分析结果转换为交互式HTML报告
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from jinja2 import Environment, FileSystemLoader, Template
import numpy as np


class ReportGenerator:
    """
    报告生成器：生成交互式HTML报告
    """

    def __init__(self, template_dir: Optional[str] = None):
        """初始化报告生成器

        Args:
            template_dir: 模板目录路径
        """
        if template_dir is None:
            template_dir = os.path.join(os.path.dirname(__file__), "templates")

        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=True
        )

        # 添加自定义过滤器和全局函数
        self.env.filters['formatNumber'] = self._format_number
        self.env.filters['formatPercent'] = self._format_percent
        self.env.globals['formatNumber'] = self._format_number
        self.env.globals['formatPercent'] = self._format_percent

    def _format_number(self, num: float) -> str:
        """格式化数字"""
        if num >= 1_000_000:
            return f"{num/1_000_000:.1f}M"
        elif num >= 1_000:
            return f"{num/1_000:.1f}K"
        else:
            return str(int(num))

    def _format_percent(self, num: float) -> str:
        """格式化百分比"""
        return f"{num * 100:.1f}%"

    def generate_overview(
        self,
        analysis_results: Dict[str, Any],
        output_path: str,
        title: str = "模型分析报告"
    ) -> str:
        """生成概览页面

        Args:
            analysis_results: 分析结果数据
            output_path: 输出HTML文件路径
            title: 报告标题

        Returns:
            生成的HTML内容
        """
        template = self.env.get_template("overview.html")

        # 准备数据
        data = self._prepare_overview_data(analysis_results)

        # 渲染模板
        html = template.render(
            title=title,
            subtitle="揭开模型黑盒，让AI白化与透明化",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data=json.dumps(data, ensure_ascii=False, indent=2),
            **data
        )

        # 保存文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        return html

    def _prepare_overview_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """准备概览页面数据

        Args:
            results: 原始分析结果

        Returns:
            格式化后的数据
        """
        model_info = results.get('model', {})

        # 关键发现
        findings = self._extract_key_findings(results)

        # 各层任务分析
        layer_tasks = self._prepare_layer_tasks(results)

        # 知识演化数据
        knowledge_evolution = self._prepare_knowledge_evolution(results)

        # 常见问题
        faqs = self._prepare_faqs(results)

        return {
            'model': model_info,
            'findings': findings,
            'layer_tasks': layer_tasks,
            'knowledge_evolution': knowledge_evolution,
            'faqs': faqs
        }

    def _extract_key_findings(self, results: Dict[str, Any]) -> List[Dict[str, str]]:
        """提取关键发现

        Args:
            results: 分析结果

        Returns:
            关键发现列表
        """
        findings = []

        # 从探针结果中提取
        probe_results = results.get('probe_results', {})

        # 找出表现最好的层
        if 'layer_accuracies' in probe_results:
            layer_accs = probe_results['layer_accuracies']
            best_layer = max(layer_accs.items(), key=lambda x: x[1])

            findings.append({
                'icon': '🎯',
                'title': f'第{best_layer[0]}层最关键',
                'description': f'该层在主要任务上达到{self._format_percent(best_layer[1])}的准确率',
                'detail': '知识在这一层已经高度抽象化，是理解模型的关键'
            })

        # 找出知识类型分布
        if 'knowledge_types' in probe_results:
            types = probe_results['knowledge_types']
            dominant_type = max(types.items(), key=lambda x: x[1])

            findings.append({
                'icon': '💡',
                'title': f'{dominant_type[0]}知识为主',
                'description': f'模型主要编码{dominant_type[0]}相关信息，占比{self._format_percent(dominant_type[1])}',
                'detail': '这反映了模型的学习目标和训练数据的特点'
            })

        # 从注意力分析中提取
        attention_results = results.get('attention_results', {})
        if 'head_patterns' in attention_results:
            patterns = attention_results['head_patterns']

            findings.append({
                'icon': '👁️',
                'title': f'发现{len(patterns)}种注意力模式',
                'description': '不同的注意力头专注于不同的语言现象',
                'detail': '这种分工合作使模型能够同时处理多种类型的信息'
            })

        return findings

    def _prepare_layer_tasks(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """准备各层任务数据

        Args:
            results: 分析结果

        Returns:
            任务数据列表
        """
        tasks = []

        probe_results = results.get('probe_results', {})

        # 定义常见任务
        task_definitions = {
            'syntax': {
                'name': '语法分析',
                'explanation': '判断词性、句法结构等语言形式特征'
            },
            'semantic': {
                'name': '语义理解',
                'explanation': '理解词义、句义等语言内容'
            },
            'reasoning': {
                'name': '逻辑推理',
                'explanation': '进行因果推断、类比推理等高级认知'
            }
        }

        # 为每个任务准备数据
        for task_key, task_def in task_definitions.items():
            if f'{task_key}_by_layer' in probe_results:
                layers_data = probe_results[f'{task_key}_by_layer']

                task_info = {
                    'name': task_def['name'],
                    'explanation': task_def['explanation'],
                    'layers': [
                        {
                            'index': i,
                            'accuracy': acc
                        }
                        for i, acc in enumerate(layers_data)
                    ]
                }

                tasks.append(task_info)

        return tasks

    def _prepare_knowledge_evolution(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """准备知识演化数据

        Args:
            results: 分析结果

        Returns:
            演化数据
        """
        probe_results = results.get('probe_results', {})

        # 默认数据（示例）
        evolution = {
            'layers': list(range(1, 13)),
            'syntax': [0.65, 0.72, 0.78, 0.82, 0.85, 0.87, 0.88, 0.89, 0.90, 0.90, 0.91, 0.91],
            'semantic': [0.45, 0.52, 0.58, 0.64, 0.70, 0.75, 0.80, 0.84, 0.87, 0.89, 0.91, 0.92],
            'reasoning': [0.35, 0.40, 0.45, 0.50, 0.56, 0.62, 0.68, 0.74, 0.80, 0.85, 0.88, 0.90]
        }

        # 如果有实际数据，替换默认值
        if 'knowledge_evolution' in probe_results:
            evolution.update(probe_results['knowledge_evolution'])

        return evolution

    def _prepare_faqs(self, results: Dict[str, Any]) -> List[Dict[str, str]]:
        """准备常见问题

        Args:
            results: 分析结果

        Returns:
            FAQ列表
        """
        return [
            {
                'question': '如何理解"层"的概念？',
                'answer': '可以把层想象成处理信息的"步骤"。信息从第1层输入，经过12层的逐步处理，每一层都会提取和组合更高级的特征。'
            },
            {
                'question': '为什么有些层准确率高，有些低？',
                'answer': '不同层专门处理不同类型的信息。早期层处理简单特征（如词性），后期层处理复杂概念。如果在某项任务上所有层都不高，说明模型可能没有学会这个任务。'
            },
            {
                'question': '注意力机制是什么？',
                'answer': '注意力机制让模型能够"关注"输入中的重要部分。就像阅读时，我们会重点关注某些关键词一样，模型通过注意力权重来决定哪些信息更重要。'
            },
            {
                'question': '这些分析结果如何帮助改进模型？',
                'answer': '通过分析可以：1) 发现模型的弱点（哪层学得不好）；2) 理解模型如何做决策；3) 指导模型压缩和优化；4) 提高模型的可解释性和可信度。'
            }
        ]

    def generate_full_report(
        self,
        analysis_results: Dict[str, Any],
        output_dir: str,
        title: str = "模型分析报告"
    ) -> str:
        """生成完整的多页面报告

        Args:
            analysis_results: 分析结果数据
            output_dir: 输出目录
            title: 报告标题

        Returns:
            主页面文件路径
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 生成概览页面
        overview_path = os.path.join(output_dir, "index.html")
        self.generate_overview(
            analysis_results,
            overview_path,
            title
        )

        # TODO: 生成其他页面
        # layers_path = self.generate_layers_page(...)
        # attention_path = self.generate_attention_page(...)
        # ...

        return overview_path


class QuickReportGenerator:
    """
    快速报告生成器：用于快速生成简化的分析报告
    """

    def __init__(self):
        self.template = """
# {{ title }}

## 关键发现

{% for finding in findings %}
### {{ finding.icon }} {{ finding.title }}

{{ finding.description }}

{% if finding.detail %}
**这意味着：** {{ finding.detail }}
{% endif %}

{% endfor %}

## 模型信息

- **层数**: {{ num_layers }}
- **参数量**: {{ num_params }}
- **隐藏维度**: {{ hidden_dim }}

## 各层表现

{% for layer in layers %}
**第{{ layer.index }}层**: {{ layer.accuracy }}%

{% endfor %}

---

*生成时间: {{ timestamp }}*
"""

    def generate(
        self,
        results: Dict[str, Any],
        output_path: str,
        title: str = "快速分析报告"
    ) -> str:
        """生成快速报告

        Args:
            results: 分析结果
            output_path: 输出路径
            title: 报告标题

        Returns:
            生成的报告内容
        """
        from jinja2 import Template

        template = Template(self.template)

        # 简化数据准备
        findings = [
            {
                'icon': '✓',
                'title': '分析完成',
                'description': f'共分析了 {results.get("num_layers", 12)} 层',
                'detail': '详细结果请查看完整报告'
            }
        ]

        content = template.render(
            title=title,
            findings=findings,
            num_layers=results.get('num_layers', 12),
            num_params=results.get('num_params', 'N/A'),
            hidden_dim=results.get('hidden_dim', 'N/A'),
            layers=results.get('layers', []),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return content

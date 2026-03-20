#!/usr/bin/env python3
"""
演示：生成交互式分析报告

展示如何使用 Model Probe 的报告系统生成人类友好的分析报告
"""

import sys
sys.path.insert(0, "/mnt/hdsd1/guomin/projects/model_probe")

from model_probe.reporting import ReportGenerator
import json


def demo_generate_report():
    """演示报告生成流程"""

    print("=" * 60)
    print("演示：生成交互式分析报告")
    print("=" * 60)

    # 步骤1：准备分析结果数据
    print("\n步骤1：准备分析结果数据...")

    # 模拟分析结果（实际使用时，这些数据来自真实的模型分析）
    analysis_results = {
        'model': {
            'name': 'GPT-2 (Small)',
            'num_layers': 12,
            'num_params': 124000000,
            'hidden_dim': 768,
            'num_heads': 12,
            'vocab_size': 50257
        },
        'probe_results': {
            'layer_accuracies': {
                6: 0.92,
                7: 0.94,
                8: 0.93
            },
            'knowledge_types': {
                '语法': 0.35,
                '语义': 0.45,
                '推理': 0.20
            },
            'syntax_by_layer': [
                0.65, 0.72, 0.78, 0.82, 0.85, 0.87,
                0.88, 0.89, 0.90, 0.90, 0.91, 0.91
            ],
            'semantic_by_layer': [
                0.45, 0.52, 0.58, 0.64, 0.70, 0.75,
                0.80, 0.84, 0.87, 0.89, 0.91, 0.92
            ],
            'reasoning_by_layer': [
                0.35, 0.40, 0.45, 0.50, 0.56, 0.62,
                0.68, 0.74, 0.80, 0.85, 0.88, 0.90
            ]
        },
        'attention_results': {
            'head_patterns': [
                '关注句首',
                '关注前一个词',
                '关注相邻词',
                '关注指代词',
                '关注疑问词'
            ]
        }
    }

    print("✓ 分析结果数据准备完成")

    # 步骤2：创建报告生成器
    print("\n步骤2：创建报告生成器...")
    generator = ReportGenerator()
    print("✓ 报告生成器创建完成")

    # 步骤3：生成概览页面
    print("\n步骤3：生成交互式HTML报告...")

    output_path = "outputs/interactive_report/index.html"
    generator.generate_overview(
        analysis_results=analysis_results,
        output_path=output_path,
        title="GPT-2 模型分析报告"
    )

    print(f"✓ 报告生成完成：{output_path}")

    # 步骤4：在浏览器中打开
    print("\n步骤4：在浏览器中打开报告...")
    print(f"请用浏览器打开文件: {output_path}")
    print(f"或访问: file://{os.path.abspath(output_path)}")

    print("\n" + "=" * 60)
    print("报告生成演示完成！")
    print("=" * 60)

    print("\n📊 报告特点：")
    print("  • 分层展示：从概览到细节")
    print("  • 交互式图表：可以hover查看详情")
    print("  • 问题驱动：回答关键问题")
    print("  • 故事化叙述：引导用户理解")
    print("  • 单文件：无需服务器，可直接在浏览器打开")

    return output_path


def demo_full_workflow():
    """演示完整的工作流程"""

    print("\n" + "=" * 60)
    print("完整工作流程演示")
    print("=" * 60)

    print("""
完整的工作流程：

1. 运行模型分析
   from model_probe import ModelWrapper, ModelConfig
   from model_probe.probes import LinearProbe

   wrapper = ModelWrapper(ModelConfig("gpt2"))
   wrapper.load_model()

   # 运行各种分析...
   results = analyzer.analyze_all()

2. 生成交互式报告
   from model_probe.reporting import ReportGenerator

   generator = ReportGenerator()
   report_path = generator.generate_full_report(
       results,
       output_dir="outputs/interactive_report",
       title="我的模型分析报告"
   )

3. 在浏览器中查看
   - 打开 index.html
   - 点击不同部分探索
   - 交互式图表可hover查看详情

4. 分享报告
   - 整个报告文件夹可以分享
   - 接收者无需安装任何依赖
   - 直接用浏览器打开即可查看
    """)


if __name__ == "__main__":
    import os

    # 运行演示
    report_path = demo_generate_report()

    # 显示完整工作流程
    demo_full_workflow()

    print("\n💡 提示：")
    print("  1. 用浏览器打开生成的报告文件")
    print("  2. 尝试hover在图表上查看详情")
    print("  3. 点击进度条查看各层详细信息")
    print("  4. 展开常见问题查看答案")

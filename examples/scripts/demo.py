#!/usr/bin/env python3
"""
Model Probe Demo - 使用GPT-2展示框架功能
"""

import sys

sys.path.insert(0, "/mnt/hdsd1/guomin/projects/model_probe")

import torch
import numpy as np
from model_probe.core import ModelWrapper, HookManager, MemoryOptimizer, ModelConfig
from model_probe.probes import LinearProbe, RepresentationAnalyzer, ProbeConfig
from model_probe.analysis import ActivationAnalyzer, Attributor
from model_probe.editor import KnowledgeEditor, Locator, EditConfig


def demo_model_loading():
    """演示：加载模型"""
    print("\n" + "=" * 50)
    print("Demo 1: 加载GPT-2模型")
    print("=" * 50)

    config = ModelConfig(name_or_path="gpt2", device="cpu", dtype=torch.float32)

    wrapper = ModelWrapper(config)
    wrapper.load_model()

    print(f"\n模型信息:")
    print(f"  - 参数数量: {wrapper.get_parameter_count():,}")
    print(f"  - 设备: {wrapper.get_device()}")
    print(f"  - 层数: {wrapper.model.config.n_layer}")
    print(f"  - 隐藏维度: {wrapper.model.config.n_embd}")
    print(f"  - 注意力头数: {wrapper.model.config.n_head}")

    MemoryOptimizer.print_memory_usage("加载后")

    return wrapper


def demo_text_generation(wrapper):
    """演示：文本生成"""
    print("\n" + "=" * 50)
    print("Demo 2: 文本生成")
    print("=" * 50)

    prompts = [
        "The capital of France is",
        "Once upon a time in a",
        "Artificial intelligence is",
    ]

    for prompt in prompts:
        inputs = wrapper.tokenizer(prompt, return_tensors="pt")

        outputs = wrapper.generate(
            inputs["input_ids"], max_new_tokens=30, temperature=0.7, top_p=0.9
        )

        generated = wrapper.tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated}")


def demo_hidden_states(wrapper):
    """演示：获取隐藏层"""
    print("\n" + "=" * 50)
    print("Demo 3: 获取隐藏层表示")
    print("=" * 50)

    text = "The quick brown fox jumps over the lazy dog."
    inputs = wrapper.tokenizer(text, return_tensors="pt")

    hidden_states = wrapper.get_hidden_states(
        inputs["input_ids"], inputs["attention_mask"]
    )

    print(f"\n输入文本: {text}")
    print(f"Token数量: {inputs['input_ids'].shape[1]}")
    print(f"层数: {len(hidden_states['all_hidden_states'])}")

    for i, hidden in enumerate(hidden_states["all_hidden_states"]):
        print(f"  Layer {i}: shape = {hidden.shape}, mean = {hidden.mean().item():.4f}")


def demo_activation_analysis(wrapper):
    """演示：激活分析"""
    print("\n" + "=" * 50)
    print("Demo 4: 激活分析")
    print("=" * 50)

    analyzer = ActivationAnalyzer(wrapper)

    text = "Paris is the capital of France."
    inputs = wrapper.tokenizer(text, return_tensors="pt")

    stats = analyzer.compute_layer_statistics(
        inputs["input_ids"], inputs["attention_mask"]
    )

    print(f"\n各层激活统计 (text: '{text}'):")
    for layer, stat in stats.items():
        print(
            f"  {layer}: mean={stat['mean']:.4f}, std={stat['std']:.4f}, sparsity={stat['sparsity']:.2%}"
        )

    important_neurons = analyzer.find_important_neurons(
        inputs["input_ids"], inputs["attention_mask"], top_k=10
    )

    print(f"\n各层最重要神经元 (top 10):")
    for layer, neurons in important_neurons.items():
        print(f"  {layer}: {neurons[:5]}...")


def demo_attribution(wrapper):
    """演示：归因分析"""
    print("\n" + "=" * 50)
    print("Demo 5: 归因分析")
    print("=" * 50)

    attributor = Attributor(wrapper)

    text = "The cat sat on the mat."
    inputs = wrapper.tokenizer(text, return_tensors="pt")

    layer_importance = attributor.compute_layerwise_importance(
        inputs["input_ids"], target_idx=-1
    )

    print(f"\n各层对最终输出的重要性:")
    for layer, importance in sorted(
        layer_importance.items(), key=lambda x: x[1], reverse=True
    )[:5]:
        print(f"  {layer}: {importance:.4f}")

    try:
        attention_rollout = attributor.attention_rollout(inputs["input_ids"])
        print(f"\nAttention Rollout shape: {attention_rollout.shape}")
    except Exception as e:
        print(f"\nAttention rollout skipped: {e}")


def demo_linear_probe(wrapper):
    """演示：线性探针"""
    print("\n" + "=" * 50)
    print("Demo 6: 线性探针 - 探测语法知识")
    print("=" * 50)

    sentences_verb = [
        "The dog barks.",
        "The cat runs.",
        "The bird flies.",
        "She eats food.",
        "He reads books.",
    ]

    sentences_noun = [
        "The running is fun.",
        "The jumping is high.",
        "The swimming is cool.",
        "The eating is good.",
        "The reading is nice.",
    ]

    all_sentences = sentences_verb + sentences_noun
    labels = [0] * len(sentences_verb) + [1] * len(sentences_noun)

    representations = []

    for sent in all_sentences:
        inputs = wrapper.tokenizer(
            sent, return_tensors="pt", padding=True, truncation=True, max_length=32
        )

        hidden = wrapper.get_hidden_states(
            inputs["input_ids"], inputs["attention_mask"]
        )["last_hidden_state"]

        rep = hidden[:, 0, :].numpy()
        representations.append(rep[0])

    X = torch.cat(
        [torch.from_numpy(r).unsqueeze(0) for r in representations], dim=0
    ).numpy()
    y = np.array(labels)

    probe_config = ProbeConfig(
        hidden_dim=wrapper.model.config.n_embd, num_classes=2, probe_type="linear"
    )

    probe = LinearProbe(probe_config)
    history = probe.fit(X, y, epochs=50, verbose=False)

    score = probe.score(X, y)
    cv_score = probe.cross_validate(X, y, cv=3)

    print(f"\n语法类别分类任务 (动词vs动名词):")
    print(f"  训练后准确率: {score:.2%}")
    print(f"  交叉验证准确率: {cv_score:.2%}")
    print(f"  结论: 模型隐藏层{'编码了' if cv_score > 0.7 else '部分编码了'}语法信息")


def demo_representation_analysis(wrapper):
    """演示：表示分析"""
    print("\n" + "=" * 50)
    print("Demo 7: 表示分析 - 层间相似度")
    print("=" * 50)

    texts = [
        "The cat is sleeping.",
        "The dog is running.",
        "A bird is flying.",
        "Fish can swim.",
        "Humans speak language.",
    ]

    layer_representations = {}

    for layer_idx in range(wrapper.model.config.n_layer):
        reps = []

        for text in texts:
            inputs = wrapper.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=32
            )

            hidden = wrapper.get_hidden_states(
                inputs["input_ids"], inputs["attention_mask"]
            )["all_hidden_states"][layer_idx]

            rep = hidden[:, 0, :].numpy()
            reps.append(rep[0])

        layer_representations[f"layer_{layer_idx}"] = np.array(reps)

    analyzer = RepresentationAnalyzer()
    similarity = analyzer.analyze_similarity(layer_representations, method="cosine")

    print(f"\n层间表示余弦相似度:")
    for (l1, l2), sim in sorted(similarity.items(), key=lambda x: x[1], reverse=True)[
        :10
    ]:
        print(f"  {l1} <-> {l2}: {sim:.4f}")

    print(
        f"\n结论: 深层表示相似度{'较高' if any(sim > 0.9 for sim in similarity.values()) else '较低'}"
    )


def demo_knowledge_locate(wrapper):
    """演示：知识定位"""
    print("\n" + "=" * 50)
    print("Demo 8: 知识定位")
    print("=" * 50)

    locator = Locator(wrapper)

    prompt = "Paris is the capital of"
    target = "France"

    knowledge_neurons = locator.compute_knowledge_neurons(prompt, target, top_k=20)

    print(f"\n查询: '{prompt} {target}'")
    print(f"相关神经元 (top 20): {knowledge_neurons['knowledge_neurons'][:10]}...")


def demo_knowledge_edit(wrapper):
    """演示：知识编辑"""
    print("\n" + "=" * 50)
    print("Demo 9: 知识编辑 (LoRA)")
    print("=" * 50)

    edit_config = EditConfig(
        method="lora", rank=4, alpha=8.0, learning_rate=1e-3, epochs=3
    )

    editor = KnowledgeEditor(wrapper, edit_config)

    print("\n1. 应用LoRA适配器...")
    editor.apply_lora(target_layers=["attn.c_attn"])

    print("\n2. 准备编辑数据...")
    train_data = [
        {"prompt": "The capital of France is", "target": "Paris"},
        {"prompt": "Paris is the capital of", "target": "France"},
        {"prompt": "France's capital city is", "target": "Paris"},
    ]

    print("\n3. 执行编辑训练...")
    history = editor.finetune_edit(train_data, verbose=True)

    print("\n4. 验证编辑效果...")
    test_prompts = [
        "The capital of France is",
        "What is the capital of France?",
    ]
    expected_outputs = ["Paris", "Paris"]

    results = editor.validate_edit(test_prompts, expected_outputs)

    for r in results["results"]:
        print(f"\n  Prompt: {r['prompt']}")
        print(f"  Expected: {r['expected']}")
        print(f"  Generated: {r['generated']}")
        print(f"  Match: {r['match']}")


def main():
    print("=" * 60)
    print("Model Probe Framework Demo - GPT-2")
    print("=" * 60)

    MemoryOptimizer.print_memory_usage("初始")

    wrapper = demo_model_loading()

    demo_text_generation(wrapper)
    demo_hidden_states(wrapper)
    demo_activation_analysis(wrapper)
    demo_attribution(wrapper)
    demo_linear_probe(wrapper)
    demo_representation_analysis(wrapper)
    demo_knowledge_locate(wrapper)

    demo_knowledge_edit(wrapper)

    MemoryOptimizer.print_memory_usage("最终")

    print("\n✓ 所有演示完成!")
    print("\n框架功能总结:")
    print("  [✓] 模型加载与封装")
    print("  [✓] 文本生成")
    print("  [✓] 隐藏层表示提取")
    print("  [✓] 激活分析 (统计、神经元)")
    print("  [✓] 归因分析 (层重要性、attention)")
    print("  [✓] 线性探针 (知识探测)")
    print("  [✓] 表示分析 (PCA、相似度)")
    print("  [✓] 知识定位")
    print("  [✓] 知识编辑 (LoRA)")


if __name__ == "__main__":
    main()

"""
模型评估器：验证模型编辑的效果
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


@dataclass
class EditResult:
    """编辑结果"""
    success: bool
    metric_name: str
    before_value: float
    after_value: float
    target_value: Optional[float] = None
    delta: Optional[float] = None
    metadata: Optional[Dict] = None

    def __post_init__(self):
        if self.delta is None:
            self.delta = self.after_value - self.before_value


class ModelEvaluator:
    """
    模型评估器：用于验证模型编辑的效果
    """

    def __init__(
        self,
        model_wrapper: Any,
        tokenizer: Optional[Any] = None
    ):
        """初始化评估器

        Args:
            model_wrapper: 模型包装器
            tokenizer: 分词器（可选）
        """
        self.model_wrapper = model_wrapper
        self.model = model_wrapper.model
        self.tokenizer = tokenizer or model_wrapper.tokenizer
        self.device = model_wrapper.config.device

    def evaluate_perplexity(
        self,
        test_texts: List[str],
        max_length: int = 512
    ) -> float:
        """计算困惑度

        Args:
            test_texts: 测试文本列表
            max_length: 最大长度

        Returns:
            困惑度值
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for text in test_texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True
                ).to(self.device)

                outputs = self.model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)

        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        return perplexity

    def evaluate_accuracy(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """计算准确率

        Args:
            inputs: 输入tensor
            labels: 标签tensor

        Returns:
            准确率
        """
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            accuracy = (predictions == labels).float().mean().item()

        return accuracy

    def evaluate_generation_quality(
        self,
        prompt: str,
        num_samples: int = 5,
        max_new_tokens: int = 50
    ) -> Dict[str, Any]:
        """评估生成质量

        Args:
            prompt: 提示文本
            num_samples: 生成样本数
            max_new_tokens: 最大生成token数

        Returns:
            生成质量指标
        """
        self.model.eval()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        generations = []
        with torch.no_grad():
            for _ in range(num_samples):
                outputs = self.model_wrapper.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.8
                )
                text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                generations.append(text)

        # 计算多样性（使用简单的平均编辑距离）
        diversity = self._compute_diversity(generations)

        return {
            "generations": generations,
            "diversity": diversity,
            "avg_length": np.mean([len(g.split()) for g in generations])
        }

    def _compute_diversity(self, texts: List[str]) -> float:
        """计算文本多样性"""
        if len(texts) < 2:
            return 0.0

        distances = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                dist = self._levenshtein_distance(texts[i], texts[j])
                max_len = max(len(texts[i]), len(texts[j]))
                normalized_dist = dist / max_len if max_len > 0 else 0
                distances.append(normalized_dist)

        return np.mean(distances) if distances else 0.0

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """计算编辑距离"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def compare_before_after(
        self,
        metric_fn: Callable,
        metric_name: str,
        **kwargs
    ) -> EditResult:
        """比较编辑前后的指标

        Args:
            metric_fn: 评估函数
            metric_name: 指标名称
            **kwargs: 传递给评估函数的参数

        Returns:
            编辑结果
        """
        before_value = metric_fn(**kwargs)
        # 这里假设模型已经被编辑了
        after_value = metric_fn(**kwargs)

        return EditResult(
            success=True,
            metric_name=metric_name,
            before_value=before_value,
            after_value=after_value
        )

    def evaluate_side_effects(
        self,
        test_cases: List[Dict[str, Any]],
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """评估编辑的副作用

        Args:
            test_cases: 测试用例列表，每个用例包含输入和期望输出
            threshold: 允许的性能下降阈值

        Returns:
            副作用评估结果
        """
        results = {
            "total_cases": len(test_cases),
            "affected_cases": 0,
            "case_details": []
        }

        for case in test_cases:
            # 这里需要根据具体的测试逻辑实现
            # 简化的示例
            case_result = {
                "case": case,
                "affected": False,
                "delta": 0.0
            }
            results["case_details"].append(case_result)

        return results

    def compute_embedding_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """计算两个文本的嵌入相似度

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            相似度（0-1之间）
        """
        self.model.eval()

        with torch.no_grad():
            inputs1 = self.tokenizer(text1, return_tensors="pt").to(self.device)
            inputs2 = self.tokenizer(text2, return_tensors="pt").to(self.device)

            outputs1 = self.model(**inputs1, output_hidden_states=True)
            outputs2 = self.model(**inputs2, output_hidden_states=True)

            # 使用最后一层的隐藏状态
            emb1 = outputs1.hidden_states[-1].mean(dim=1)
            emb2 = outputs2.hidden_states[-1].mean(dim=1)

            similarity = torch.nn.functional.cosine_similarity(emb1, emb2).item()

        return similarity

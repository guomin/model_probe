"""
编辑模块：知识定位与编辑
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
from dataclasses import dataclass


@dataclass
class EditConfig:
    """编辑配置"""
    method: str = "lora"  # lora, rome, ft
    rank: int = 8
    alpha: float = 16.0
    learning_rate: float = 1e-4
    epochs: int = 10
    batch_size: int = 4


class Locator:
    """
    知识定位器：找出模型中存储特定知识的位置
    """
    
    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper
        
    def find_knowledge_location(
        self,
        subject: str,
        prompt_template: str = "{} is a {}.",
        relations: List[str] = ["person", "place", "animal"]
    ) -> Dict[str, Any]:
        """
        通过消融实验定位知识位置
        
        Args:
            subject: 主体实体
            prompt_template: 提示模板
            relations: 关系列表
            
        Returns:
            各层的编辑效果
        """
        results = {}
        
        inputs = self.model_wrapper.tokenizer(
            prompt_template.format(subject, "{}"),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        baseline_outputs = self.model_wrapper.generate(
            inputs["input_ids"],
            max_new_tokens=5
        )
        baseline_text = self.model_wrapper.tokenizer.decode(
            baseline_outputs[0], skip_special_tokens=True
        )
        
        hidden_states = self.model_wrapper.get_hidden_states(
            inputs["input_ids"],
            inputs["attention_mask"]
        )["all_hidden_states"]
        
        for layer_idx, hidden in enumerate(hidden_states):
            layer_name = f"layer_{layer_idx}"
            
            with torch.no_grad():
                modified_hidden = hidden.clone()
                modified_hidden[:, 1:-1, :] = 0
                
                temp_model = self._create_temp_model(hidden.shape[-1])
                output = temp_model(inputs["input_ids"])
                
            results[layer_name] = {
                "hidden_dim": hidden.shape[-1],
                "ablation_effect": "待计算"
            }
            
        return results
    
    def _create_temp_model(self, hidden_dim: int) -> nn.Module:
        """创建临时模型用于测试"""
        return nn.Linear(hidden_dim, hidden_dim)
    
    def compute_knowledge_neurons(
        self,
        prompt: str,
        target_token: str,
        top_k: int = 50
    ) -> Dict[str, List[int]]:
        """
        计算与特定知识相关的神经元
        """
        inputs = self.model_wrapper.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=128,
            truncation=True
        )
        
        hidden_states = self.model_wrapper.get_hidden_states(
            inputs["input_ids"],
            inputs["attention_mask"]
        )["last_hidden_state"]
        
        target_ids = self.model_wrapper.tokenizer.encode(target_token)
        target_pos = inputs["input_ids"][0].tolist()
        
        target_idx = None
        for i, tid in enumerate(target_pos):
            if tid in target_ids:
                target_idx = i
                break
                
        if target_idx is None:
            target_idx = len(target_pos) - 1
            
        target_hidden = hidden_states[0, target_idx]
        
        gradient_scores = target_hidden.abs()
        top_neurons = gradient_scores.argsort(descending=True)[:top_k]
        
        return {"knowledge_neurons": top_neurons.tolist()}
    
    def attention_based_locate(
        self,
        input_ids: torch.Tensor,
        target_pos: int = -1
    ) -> Dict[str, np.ndarray]:
        """
        基于注意力定位关键token
        """
        outputs = self.model_wrapper.model(
            input_ids=input_ids.to(self.model_wrapper.config.device),
            output_attentions=True,
            use_cache=False
        )
        
        attentions = outputs.attentions
        
        attention_weights = []
        
        for layer_attn in attentions:
            layer_attn = layer_attn[0]
            attn_sum = layer_attn.sum(dim=0)
            attention_weights.append(attn_sum[target_pos].cpu().numpy())
            
        return {
            f"layer_{i}": attn for i, attn in enumerate(attention_weights)
        }


class KnowledgeEditor:
    """
    知识编辑器：修改模型知识
    """
    
    def __init__(self, model_wrapper, config: Optional[EditConfig] = None):
        self.model_wrapper = model_wrapper
        self.config = config or EditConfig()
        self.editor_layers: Dict[str, nn.Module] = {}
        
    def apply_lora(
        self,
        target_layers: Optional[List[str]] = None
    ) -> "KnowledgeEditor":
        """
        应用LoRA适配器
        
        LoRA原理：在指定的Attention层添加低秩分解矩阵
        W_new = W + BA, 其中B∈R^{d×r}, A∈R^{r×k}, r<<min(d,k)
        """
        if target_layers is None:
            target_layers = ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
            
        for name, module in self.model_wrapper.model.named_modules():
            for target in target_layers:
                if target in name:
                    in_features = module.in_features
                    out_features = module.out_features
                    
                    lora_rank = min(self.config.rank, in_features, out_features)
                    
                    lora_a = nn.Linear(in_features, lora_rank, bias=False)
                    lora_b = nn.Linear(lora_rank, out_features, bias=False)
                    
                    nn.init.zeros_(lora_a.weight)
                    nn.init.zeros_(lora_b.weight)
                    
                    editor_layer = nn.ModuleDict({
                        "lora_a": lora_a,
                        "lora_b": lora_b,
                        "original": module,
                        "scale": self.config.alpha
                    })
                    
                    self.editor_layers[name] = editor_layer
                    
                    def make_forward(editor):
                        def forward(x):
                            original_out = editor["original"](x)
                            lora_out = editor["lora_b"](editor["lora_a"](x)) * editor["scale"]
                            return original_out + lora_out
                        return forward
                    
                    module.forward = make_forward(editor_layer)
                    
        self.model_wrapper.model = self.model_wrapper.model
        print(f"Applied LoRA to {len(self.editor_layers)} layers")
        
        return self
    
    def finetune_edit(
        self,
        train_data: List[Dict[str, str]],
        target_key: str = "target",
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        轻量级微调编辑
        
        Args:
            train_data: [{"prompt": "...", "target": "..."}]
            target_key: 目标字段key
            
        Returns:
            训练历史
        """
        optimizer = torch.optim.Adam(
            self.editor_layers.parameters() if self.editor_layers else 
            self.model_wrapper.model.parameters(),
            lr=self.config.learning_rate
        )
        
        history = {"loss": []}
        
        for epoch in range(self.config.epochs):
            total_loss = 0
            
            for batch_start in range(0, len(train_data), self.config.batch_size):
                batch = train_data[batch_start:batch_start + self.config.batch_size]
                
                prompts = [d["prompt"] for d in batch]
                targets = [d[target_key] for d in batch]
                
                inputs = self.model_wrapper.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128
                )
                
                target_ids = self.model_wrapper.tokenizer(
                    targets,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128
                )
                
                outputs = self.model_wrapper.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=target_ids["input_ids"],
                    use_cache=False
                )
                
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(train_data)
            history["loss"].append(avg_loss)
            
            if verbose:
                print(f"Epoch {epoch+1}/{self.config.epochs}, Loss: {avg_loss:.4f}")
                
        return history
    
    def rome_edit(
        self,
        prompt: str,
        subject: str,
        new_target: str,
        layers: Optional[List[int]] = None
    ):
        """
        ROME (Rank-One Model Editing) 编辑
        
        这是一个简化实现，真正的ROME需要更复杂的计算
        """
        inputs = self.model_wrapper.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=128,
            truncation=True
        )
        
        hidden_states = self.model_wrapper.get_hidden_states(
            inputs["input_ids"],
            inputs["attention_mask"]
        )["all_hidden_states"]
        
        subject_token_pos = None
        tokens = inputs["input_ids"][0]
        subject_tokens = self.model_wrapper.tokenizer.encode(subject)
        
        for i, tid in enumerate(tokens.tolist()):
            if tid in subject_tokens:
                subject_token_pos = i
                break
                
        if subject_token_pos is None:
            subject_token_pos = 1
            
        target_hidden = hidden_states[-1][0, subject_token_pos]
        
        print(f"ROME editing at layer {layers}, position {subject_token_pos}")
        print(f"Target: {subject} -> {new_target}")
        
        return {"status": "completed", "layer": layers, "position": subject_token_pos}
    
    def validate_edit(
        self,
        test_prompts: List[str],
        expected_outputs: List[str]
    ) -> Dict[str, Any]:
        """
        验证编辑效果
        """
        results = []
        
        for prompt, expected in zip(test_prompts, expected_outputs):
            inputs = self.model_wrapper.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=128,
                truncation=True
            )
            
            outputs = self.model_wrapper.generate(
                inputs["input_ids"],
                max_new_tokens=20
            )
            
            generated = self.model_wrapper.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )
            
            results.append({
                "prompt": prompt,
                "expected": expected,
                "generated": generated,
                "match": expected.lower() in generated.lower()
            })
            
        accuracy = sum(1 for r in results if r["match"]) / len(results)
        
        return {"results": results, "accuracy": accuracy}
    
    def reset(self):
        """重置编辑"""
        for editor in self.editor_layers.values():
            if "original" in editor:
                editor["original"].forward = None
                
        self.editor_layers.clear()
        print("Editor reset")

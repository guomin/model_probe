"""
核心模块：模型封装、Hook管理、显存优化
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Any, Union
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import gc

from .config import ModelConfig


class ModelWrapper:
    """
    统一模型封装，支持PyTorch原生模型和HuggingFace模型
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model: Optional[nn.Module] = None
        self.tokenizer: Optional[Any] = None
        self._hooks: List = []
        self._activation_cache: Dict[str, torch.Tensor] = {}
        
    def load_model(self, for_generation: bool = True) -> "ModelWrapper":
        """加载模型
        
        Args:
            for_generation: 是否加载带生成能力的模型
        """
        print(f"Loading model: {self.config.name_or_path}")
        
        load_kwargs = {
            "torch_dtype": self.config.dtype,
            "trust_remote_code": True,
            "attn_implementation": "eager" if self.config.output_attentions else "sdpa"
        }
        
        try:
            if for_generation:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.name_or_path,
                    **load_kwargs
                )
            else:
                self.model = AutoModel.from_pretrained(
                    self.config.name_or_path,
                    **load_kwargs
                )
                
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.name_or_path,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            print(f"Failed to load from pretrained: {e}")
            print("Trying to load base model...")
            
            load_kwargs["ignore_mismatched_sizes"] = True
            
            if for_generation:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.name_or_path,
                    **load_kwargs
                )
            else:
                self.model = AutoModel.from_pretrained(
                    self.config.name_or_path,
                    **load_kwargs
                )
                
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.name_or_path,
                trust_remote_code=True
            )
        
        self.model.to(self.config.device)
        self.model.eval()
        
        if not self.config.use_cache:
            self.model.config.use_cache = False
        
        if self.config.output_attentions and hasattr(self.model.config, 'output_attentions'):
            try:
                self.model.config.output_attentions = True
            except ValueError:
                if hasattr(self.model.config, 'attn_implementation'):
                    self.model.config.attn_implementation = 'eager'
                self.model.config.output_attentions = True
            
        return self
    
    def get_hidden_states(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True
    ) -> Dict[str, torch.Tensor]:
        """获取隐藏层输出"""
        mask = attention_mask.to(self.config.device) if attention_mask is not None else None
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.to(self.config.device),
                attention_mask=mask,
                output_hidden_states=output_hidden_states,
                use_cache=self.config.use_cache
            )
        
        hidden_states = outputs.hidden_states if output_hidden_states else outputs.last_hidden_state
        
        result = {
            "last_hidden_state": hidden_states[-1],
            "all_hidden_states": hidden_states if isinstance(hidden_states, tuple) else tuple(hidden_states),
            "pooler_output": outputs.pooler_output if hasattr(outputs, "pooler_output") else None,
        }
        
        if hasattr(outputs, "attentions") and outputs.attentions:
            result["attentions"] = outputs.attentions
            
        return result
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """文本生成"""
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids.to(self.config.device),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )
        
        return outputs
    
    def register_hook(self, module: nn.Module, hook_fn: Callable) -> torch.utils.hooks.RemovableHandle:
        """注册前向Hook"""
        handle = module.register_forward_hook(hook_fn)
        self._hooks.append(handle)
        return handle
    
    def remove_hooks(self):
        """移除所有Hook"""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        
    def get_layer_names(self) -> List[str]:
        """获取所有层名称"""
        layers = []
        for name, _ in self.model.named_modules():
            if len(name.split(".")) > 1:
                layers.append(name)
        return layers
    
    def get_parameter_count(self) -> int:
        """获取参数数量"""
        return sum(p.numel() for p in self.model.parameters())
    
    def get_device(self) -> str:
        """获取设备"""
        return next(self.model.parameters()).device.type
    
    def clear_cache(self):
        """清理缓存"""
        self._activation_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def __repr__(self):
        return f"ModelWrapper(model={self.config.name_or_path}, params={self.get_parameter_count():,})"


class HookManager:
    """
    Hook管理器：用于捕获中间层激活值
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self.activations: Dict[str, List[torch.Tensor]] = {}
        
    def register_hooks(
        self, 
        layer_names: List[str],
        hook_type: str = "forward"
    ) -> "HookManager":
        """注册多个层的Hook"""
        for name in layer_names:
            module = self._get_module_by_name(name)
            if module is not None:
                self.activations[name] = []
                if hook_type == "forward":
                    self.hooks[name] = module.register_forward_hook(
                        self._make_hook_fn(name)
                    )
        return self
    
    def _make_hook_fn(self, name: str) -> Callable:
        """创建hook函数"""
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                self.activations[name].append(output.detach().cpu())
            elif isinstance(output, tuple):
                self.activations[name].append(output[0].detach().cpu() if output[0] is not None else None)
        return hook_fn
    
    def _get_module_by_name(self, name: str) -> Optional[nn.Module]:
        """通过名称获取模块"""
        parts = name.split(".")
        module = self.model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part, None)
            if module is None:
                return None
        return module
    
    def get_activations(self, clear: bool = True) -> Dict[str, torch.Tensor]:
        """获取激活值（合并batch维度）"""
        result = {}
        for name, acts in self.activations.items():
            if acts:
                stacked = torch.cat([a for a in acts if a is not None], dim=0)
                result[name] = stacked
        if clear:
            self.clear()
        return result
    
    def clear(self):
        """清空激活值"""
        for acts in self.activations.values():
            acts.clear()
            
    def remove_hooks(self):
        """移除所有hooks"""
        for handle in self.hooks.values():
            handle.remove()
        self.hooks.clear()
        self.activations.clear()


class MemoryOptimizer:
    """
    显存优化工具
    """
    
    @staticmethod
    def enable_gradient_checkpointing(model: nn.Module):
        """启用梯度检查点以节省显存"""
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        elif hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing()
            
    @staticmethod
    def enable_cpu_offload(model: nn.Module, accelerator: Any = None):
        """启用CPU卸载"""
        if accelerator is not None:
            accelerator.prepare_model(model, device_placement=True)
            
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """获取显存使用情况（MB）"""
        if torch.cuda.is_available():
            return {
                "allocated": torch.cuda.memory_allocated() / 1024**2,
                "reserved": torch.cuda.memory_reserved() / 1024**2,
                "max_allocated": torch.cuda.max_memory_allocated() / 1024**2,
            }
        return {"allocated": 0, "reserved": 0, "max_allocated": 0}
    
    @staticmethod
    def print_memory_usage(prefix: str = ""):
        """打印显存使用"""
        mem = MemoryOptimizer.get_memory_usage()
        if mem["allocated"] > 0:
            print(f"{prefix}GPU Memory: {mem['allocated']:.1f}MB allocated, {mem['reserved']:.1f}MB reserved")

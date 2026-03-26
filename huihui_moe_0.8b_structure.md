# Huihui-MoE-0.8B-2E 模型结构分析

## 模型基本信息

- **架构**: Qwen3MoeForCausalLM
- **总参数量**: 0.86B
- **数据类型**: bfloat16
- **词汇量**: 151936
- **最大上下文长度**: 40960

## 核心配置

| 参数 | 值 |
|------|-----|
| hidden_size | 1024 |
| num_hidden_layers | 28 |
| num_attention_heads | 16 |
| num_key_value_heads | 8 |
| head_dim | 128 |
| intermediate_size | 3072 |
| moe_intermediate_size | 3072 |
| num_experts | 2 |
| num_experts_per_tok | 1 |

## 每层结构组成 (共28层)

每层包含以下组件：

### 1. Self-Attention (自注意力)
| 组件 | 形状 | 参数量 |
|------|------|--------|
| q_proj | [2048, 1024] | 2.10M |
| k_proj | [1024, 1024] | 1.05M |
| v_proj | [1024, 1024] | 1.05M |
| o_proj | [1024, 2048] | 2.10M |
| q_norm | [128] | 128 |
| k_norm | [128] | 128 |

### 2. MoE MLP (混合专家)
| 组件 | 形状 | 参数量 |
|------|------|--------|
| experts.0.gate_proj | [3072, 1024] | 3.15M |
| experts.0.up_proj | [3072, 1024] | 3.15M |
| experts.0.down_proj | [1024, 3072] | 3.15M |
| experts.1.gate_proj | [3072, 1024] | 3.15M |
| experts.1.up_proj | [3072, 1024] | 3.15M |
| experts.1.down_proj | [1024, 3072] | 3.15M |
| gate (router) | [2, 1024] | 2.05K |

### 3. LayerNorm
| 组件 | 形状 | 参数量 |
|------|------|--------|
| input_layernorm | [1024] | 1.02K |
| post_attention_layernorm | [1024] | 1.02K |

## 参数量统计

| 组件 | 参数量 | 占比 |
|------|--------|------|
| Embedding | 155.58M | 18.1% |
| Attention | 176.17M | 20.5% |
| MoE Experts | 528.48M | 61.4% |
| Router Gate | 57.34K | 0.01% |
| LayerNorm | 57.34K | 0.01% |
| **总计** | **0.86B** | 100% |

## MoE 专家机制详解

### 每层专家结构
```
Layer N:
  ├── Expert 0: gate_proj + up_proj + down_proj (SwiGLU)
  └── Expert 1: gate_proj + up_proj + down_proj (SwiGLU)
  └── Router Gate: 决定激活哪个 Expert
```

### 工作原理
1. **Router (gate)** 计算每个 token 应该路由到哪个 Expert
2. **Top-1 路由**: 每个 token 只激活1个 Expert（2选1）
3. 未激活的 Expert 参数不参与前向计算

### 稀疏 MoE 特性
- **专家总数**: 28层 × 2专家 = 56个专家模块
- **激活比例**: 每次推理只激活 50% 的 FFN 参数
- **活跃参数量**: 528.48M / 2 = 264.24M / 层

## 总结

Huihui-MoE-0.8B-2E 是一个基于 Qwen3-MoE 架构的稀疏 MoE 模型：
- 通过 2 个 Experts 的设计，在 0.86B 总参数量下实现了高效的推理
- 每个 token 只激活 1 个 Expert，兼顾了模型容量和计算效率
- 采用 SwiGLU 激活函数和 RMSNorm 归一化
- 支持超长上下文 (40960 tokens)

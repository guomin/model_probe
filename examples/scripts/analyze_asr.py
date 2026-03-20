#!/usr/bin/env python3
"""
ASR模型内部表示分析 - 使用Whisper/Wav2Vec
"""

import sys
sys.path.insert(0, "/mnt/hdsd1/guomin/projects/model_probe")

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model_probe.probes import LinearProbe, ProbeConfig, RepresentationAnalyzer


def analyze_wav2vec():
    """分析Wav2Vec 2.0模型"""
    print("\n" + "="*60)
    print("ASR模型分析: Wav2Vec 2.0 Base")
    print("="*60)
    
    try:
        from transformers import Wav2Vec2Model, Wav2Vec2Processor
    except ImportError:
        print("需要安装transformers新版本")
        return
    
    print("\n加载Wav2Vec 2.0 Base模型...")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    
    model.eval()
    
    print(f"\n模型信息:")
    print(f"  - 参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - 层数: {model.config.num_hidden_layers}")
    print(f"  - 隐藏维度: {model.config.hidden_size}")
    print(f"  - 注意力头数: {model.config.num_attention_heads}")
    
    return model, processor


def analyze_whisper():
    """分析Whisper模型"""
    print("\n" + "="*60)
    print("ASR模型分析: Whisper Tiny")
    print("="*60)
    
    try:
        from transformers import WhisperModel, WhisperProcessor
    except ImportError:
        print("需要安装transformers新版本")
        return
    
    print("\n加载Whisper Tiny模型...")
    model = WhisperModel.from_pretrained("openai/whisper-tiny")
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    
    model.eval()
    
    print(f"\n模型信息:")
    print(f"  - 参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Encoder层数: {model.config.encoder_layers}")
    print(f"  - Decoder层数: {model.config.decoder_layers}")
    print(f"  - 隐藏维度: {model.config.d_model}")
    print(f"  - 注意力头数: {model.config.encoder_attention_heads}")
    
    return model, processor


def probe_speaker_identity(model, processor):
    """探针：说话人识别（同一说话人 vs 不同说话人）"""
    print("\n" + "="*60)
    print("实验: 说话人身份探测")
    print("="*60)
    
    print("\n注意: 需要音频文件，这里使用模拟数据演示探针原理")
    print("实际使用时需要准备说话人标注的音频数据\n")
    
    print("Wav2Vec2 隐藏层可以用于:")
    print("  - 说话人识别 (speaker identification)")
    print("  - 语音情感识别 (emotion recognition)")
    print("  - 语言识别 (language identification)")
    print("  - 音素识别 (phoneme recognition)")
    
    return None


def analyze_encoder_vs_decoder(model, processor):
    """分析Encoder vs Decoder的表示差异"""
    print("\n" + "="*60)
    print("实验: Encoder vs Decoder 表示分析 (Whisper)")
    print("="*60)
    
    if not hasattr(model, 'encoder') or not hasattr(model, 'decoder'):
        print("模型不是encoder-decoder架构")
        return
    
    print("""
Whisper模型结构:
┌─────────────────────────────────────────────────────────────┐
│                    Whisper Architecture                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Audio Input                                                │
│      │                                                      │
│      ▼                                                      │
│  ┌─────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │Encoder │ ───▶ │ Cross Attention│ ───▶ │   Decoder    │  │
│  │(6层)   │      │               │      │   (6层)      │  │
│  └─────────┘      └──────────────┘      └──────────────┘  │
│      │                                      │              │
│      │                                      ▼              │
│  Audio                                          Text Output │
│  Features                                       (autoregressive) │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Encoder: 处理音频，生成声学表示
Decoder: 逐token生成文本，利用Encoder的表示
""")
    
    return None


def visualize_asr_architecture():
    """可视化ASR架构对比"""
    print("\n" + "="*60)
    print("ASR模型架构对比")
    print("="*60)
    
    architectures = """
┌────────────────────────────────────────────────────────────────────┐
│                    ASR模型架构对比                                  │
├──────────────────┬─────────────────────┬───────────────────────────┤
│     特性         │    Wav2Vec 2.0      │       Whisper             │
├──────────────────┼─────────────────────┼───────────────────────────┤
│ 架构             │ Encoder-only        │ Encoder-Decoder           │
│ 训练方式         │ CTC (Connectionist │ Seq2Seq                   │
│                  │ Temporal            │                           │
│                  │ Classification)     │                           │
├──────────────────┼─────────────────────┼───────────────────────────┤
│ 输出方式         │ 逐帧声学特征        │ 自回归文本生成            │
│                  │ + CTC decoder       │                           │
├──────────────────┼─────────────────────┼───────────────────────────┤
│ 参数规模         │ 94M (base)          │ 39M (tiny)                │
│                  │ 317M (large)        │ 1550M (large)            │
├──────────────────┼─────────────────────┼───────────────────────────┤
│ 优势             │ 快速推理            │ 多语言支持                │
│                  │ 精细音频表示        │ 上下文理解                │
│                  │ 适合微调            │ 端到端                    │
├──────────────────┼─────────────────────┼───────────────────────────┤
│ 典型任务         │ 说话人识别          │ 语音翻译                  │
│                  │ 语音情感            │ 语音识别                  │
│                  │ 音频分类            │ 音频描述                  │
└──────────────────┴─────────────────────┴───────────────────────────┘
"""
    print(architectures)


def demo_probe_principle():
    """演示探针在ASR中的应用原理"""
    print("\n" + "="*60)
    print("探针在ASR模型中的应用")
    print("="*60)
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    ASR模型探针分析流程                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. 准备音频数据                                                    │
│     audio_1: "hello world" (speaker A)                            │
│     audio_2: "hello world" (speaker B)                            │
│     audio_3: "how are you" (speaker A)                            │
│     ...                                                            │
│                                                                     │
│  2. 提取隐藏表示                                                    │
│     inputs = processor(audio)                                       │
│     hidden = model(inputs.input_values)                            │
│     representations = hidden.last_hidden_state                      │
│                                                                     │
│  3. 训练探针                                                        │
│     X = representations[:, 0, :]  # 取CLS或平均                    │
│     y = [0, 1, 0, ...]  # 说话人标签                              │
│                                                                     │
│     probe = LinearProbe(hidden_dim=768)                           │
│     probe.fit(X, y)                                                │
│                                                                     │
│  4. 分析结果                                                        │
│     accuracy = probe.score(X, y)                                   │
│                                                                     │
│  5. 结论                                                            │
│     高准确率 → 模型编码了说话人信息                                 │
│     低准确率 → 模型未编码说话人信息                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

常见ASR探针任务:
────────────────────────────────────────────────────────────────────
任务              标签示例                探针目标
────────────────────────────────────────────────────────────────────
说话人识别        Speaker A / B          Encoder中间层
情感识别          Happy / Sad / Angry     Encoder高层
噪声类型          室内 / 室外 / 车载      Encoder低层
语言识别          英语 / 中文 / 日语      Encoder中层
口音识别          美式 / 英式 / 澳式     Encoder高层
────────────────────────────────────────────────────────────────────
""")


def main():
    print("="*60)
    print("ASR模型内部表示分析")
    print("="*60)
    
    visualize_asr_architecture()
    
    demo_probe_principle()
    
    print("\n" + "="*60)
    print("可选: 加载真实模型进行实验")
    print("="*60)
    print("""
选项1: Wav2Vec 2.0 Base (~94M参数)
  - 需要音频文件(.wav)
  - 支持说话人识别、情感识别等任务

选项2: Whisper Tiny (~39M参数)
  - 支持多语言语音识别
  - 可分析Encoder vs Decoder表示差异

由于当前环境没有音频文件，您可以:
1. 准备自己的音频数据
2. 运行框架代码进行实际分析

代码示例:
  from transformers import Wav2Vec2Model, Wav2Vec2Processor
  import torch
  
  model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
  processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
  
  # 加载音频
  import librosa
  audio, sr = librosa.load("your_audio.wav", sr=16000)
  inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
  
  # 获取隐藏表示
  with torch.no_grad():
    outputs = model(inputs.input_values)
    hidden = outputs.last_hidden_state
  
  print(f"Hidden shape: {hidden.shape}")
""")
    
    print("\n✓ ASR分析框架说明完成!")


if __name__ == "__main__":
    main()

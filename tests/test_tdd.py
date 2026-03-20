"""
测试驱动开发 (TDD) 示例

遵循 TDD 原则：
1. 先写测试（红色）
2. 实现功能使其通过（绿色）
3. 重构优化（黄色）
"""

import pytest
import torch
import numpy as np


class TestTDDWorkflow:
    """
    TDD 示例：实现一个新的探针类型
    
    工作流程：
    1. 编写一个会失败的测试
    2. 运行测试确认失败
    3. 实现最小代码使其通过
    4. 重构优化
    """
    
    def test_mlp_probe_basic(self):
        """
        TDD Step 1: 假设我们要添加 MLPProbe
        这是一个会失败的测试，定义我们想要的行为
        """
        from model_probe.probes import LinearProbe, ProbeConfig
        
        config = ProbeConfig(hidden_dim=128, num_classes=3, probe_type="mlp")
        probe = LinearProbe(config)
        
        X = np.random.randn(50, 128)
        y = np.random.randint(0, 3, 50)
        
        probe.fit(X, y, epochs=5, verbose=False)
        
        assert probe.is_fitted
        assert probe.model is not None


class TestLayerSelectionProbe:
    """
    业务需求：需要找出编码特定信息的最优层
    """
    
    def test_find_best_layer(self):
        """找出某任务的最优层"""
        from model_probe.probes import LinearProbe, ProbeConfig
        
        n_layers = 12
        n_samples = 100
        hidden_dim = 768
        
        hidden_states = {
            f"layer_{i}": np.random.randn(n_samples, hidden_dim) 
            for i in range(n_layers)
        }
        
        y = np.random.randint(0, 2, n_samples)
        
        best_layer = None
        best_acc = 0
        
        for layer_name, X in hidden_states.items():
            probe = LinearProbe(ProbeConfig(hidden_dim=hidden_dim, num_classes=2))
            probe.fit(X, y, epochs=10, verbose=False)
            acc = probe.score(X, y)
            
            if acc > best_acc:
                best_acc = acc
                best_layer = layer_name
                
        assert best_layer is not None
        assert 0 <= best_acc <= 1


class TestIntervention:
    """
    业务需求：干预模型行为
    """
    
    def test_ablate_neurons(self):
        """测试神经元消融"""
        from model_probe.analysis import WeightAnalyzer
        import torch.nn as nn
        
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        analyzer = WeightAnalyzer(model)
        
        original_weight = model[0].weight.data.clone()
        
        stats = analyzer.compute_weight_statistics()
        
        assert "0.weight" in stats
        assert stats["0.weight"]["mean"] is not None


class TestIntegrationScenarios:
    """
    集成测试场景
    """
    
    def test_full_analysis_pipeline(self):
        """完整分析流程"""
        from model_probe.probes import LinearProbe, RepresentationAnalyzer, ProbeConfig
        from model_probe.analysis import WeightAnalyzer
        import torch.nn as nn
        
        model = nn.Linear(100, 100)
        
        analyzer = WeightAnalyzer(model)
        stats = analyzer.compute_weight_statistics()
        assert len(stats) > 0
        
        rep_analyzer = RepresentationAnalyzer()
        X = np.random.randn(30, 100)
        X_pca, _ = rep_analyzer.compute_pca(X, n_components=10)
        assert X_pca.shape[1] == 10
        
        probe = LinearProbe(ProbeConfig(hidden_dim=100, num_classes=2))
        y = np.random.randint(0, 2, 30)
        probe.fit(X, y, epochs=5, verbose=False)
        assert probe.is_fitted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

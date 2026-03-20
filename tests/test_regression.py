"""
回归测试：确保核心功能不被破坏
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch


class TestWrapperRegression:
    """ModelWrapper 回归测试"""
    
    @pytest.fixture
    def mock_model(self):
        """创建模拟模型"""
        model = MagicMock()
        model.config.use_cache = False
        model.config.output_attentions = True
        model.named_parameters.return_value = [
            ("layer.0.weight", torch.randn(768, 768)),
        ]
        return model
    
    @pytest.fixture
    def wrapper(self, mock_model):
        """创建 ModelWrapper"""
        from model_probe.core import ModelWrapper, ModelConfig
        config = ModelConfig(name_or_path="gpt2", device="cpu")
        wrapper = ModelWrapper(config)
        wrapper.model = mock_model
        return wrapper
    
    def test_wrapper_creation(self, wrapper):
        """测试 wrapper 能正常创建"""
        assert wrapper is not None
        assert isinstance(wrapper.config, type(wrapper.config))
        
    def test_get_hidden_states_shape(self, wrapper):
        """测试隐藏状态获取"""
        mock_output = MagicMock()
        mock_output.hidden_states = tuple([
            torch.randn(1, 10, 768) for _ in range(12)
        ])
        mock_output.last_hidden_state = torch.randn(1, 10, 768)
        mock_output.attentions = None
        wrapper.model.return_value = mock_output
        
        input_ids = torch.randint(0, 50000, (1, 10))
        result = wrapper.get_hidden_states(input_ids)
        
        assert "last_hidden_state" in result
        assert "all_hidden_states" in result
        assert result["last_hidden_state"].shape == (1, 10, 768)
        
    def test_generate_returns_tensor(self, wrapper):
        """测试生成返回 tensor"""
        mock_output = torch.randint(0, 50000, (1, 20))
        wrapper.model.generate.return_value = mock_output
        wrapper.tokenizer = MagicMock()
        wrapper.tokenizer.pad_token_id = 50256
        
        input_ids = torch.randint(0, 50000, (1, 10))
        result = wrapper.generate(input_ids, max_new_tokens=10)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 1
        
    def test_register_hook(self, wrapper):
        """测试 Hook 注册"""
        mock_module = MagicMock()
        mock_handle = MagicMock()
        mock_module.register_forward_hook.return_value = mock_handle
        
        handle = wrapper.register_hook(mock_module, lambda m, i, o: None)
        
        assert handle is not None
        assert len(wrapper._hooks) == 1
        
    def test_memory_operations(self, wrapper):
        """测试内存操作"""
        wrapper._activation_cache = {"key": torch.randn(10, 10)}
        wrapper.clear_cache()
        assert len(wrapper._activation_cache) == 0


class TestProbesRegression:
    """探针回归测试"""
    
    def test_linear_probe_creation(self):
        """测试线性探针创建"""
        from model_probe.probes import LinearProbe, ProbeConfig
        
        config = ProbeConfig(hidden_dim=768, num_classes=2)
        probe = LinearProbe(config)
        
        assert probe.config.hidden_dim == 768
        assert probe.config.num_classes == 2
        
    def test_linear_probe_fit_predict(self):
        """测试探针训练和预测"""
        from model_probe.probes import LinearProbe, ProbeConfig
        
        config = ProbeConfig(hidden_dim=10, num_classes=2)
        probe = LinearProbe(config)
        
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        
        probe.fit(X, y, epochs=10, verbose=False)
        
        assert probe.is_fitted
        
        pred = probe.predict(X[:5])
        assert len(pred) == 5
        
    def test_representation_analyzer(self):
        """测试表示分析器"""
        from model_probe.probes import RepresentationAnalyzer
        
        analyzer = RepresentationAnalyzer()
        
        X = np.random.randn(50, 20)
        X_pca, pca = analyzer.compute_pca(X, n_components=10)
        
        assert X_pca.shape == (50, 10)
        assert pca is not None


class TestAnalysisRegression:
    """分析模块回归测试"""
    
    def test_activation_analyzer_creation(self):
        """测试激活分析器创建"""
        from model_probe.analysis import ActivationAnalyzer
        
        mock_wrapper = MagicMock()
        analyzer = ActivationAnalyzer(mock_wrapper)
        
        assert analyzer.model_wrapper == mock_wrapper
        
    def test_attributor_creation(self):
        """测试归因分析器创建"""
        from model_probe.analysis import Attributor
        
        mock_wrapper = MagicMock()
        attr = Attributor(mock_wrapper)
        
        assert attr.model_wrapper == mock_wrapper


class TestEditorRegression:
    """编辑器回归测试"""
    
    def test_edit_config_defaults(self):
        """测试编辑配置默认值"""
        from model_probe.editor import EditConfig
        
        config = EditConfig()
        
        assert config.method == "lora"
        assert config.rank == 8
        
    def test_knowledge_editor_creation(self):
        """测试编辑器创建"""
        from model_probe.editor import KnowledgeEditor, EditConfig
        
        mock_wrapper = MagicMock()
        editor = KnowledgeEditor(mock_wrapper)
        
        assert editor.model_wrapper == mock_wrapper
        
    def test_locator_creation(self):
        """测试定位器创建"""
        from model_probe.editor import Locator
        
        mock_wrapper = MagicMock()
        locator = Locator(mock_wrapper)
        
        assert locator.model_wrapper == mock_wrapper


class TestStaticRegression:
    """静态分析回归测试"""

    def test_weight_analyzer_creation(self):
        """测试权重分析器创建"""
        from model_probe.analysis import WeightAnalyzer
        import torch.nn as nn

        model = nn.Linear(10, 10)
        analyzer = WeightAnalyzer(model)

        assert analyzer.model == model

    def test_weight_statistics(self):
        """测试权重统计"""
        from model_probe.analysis import WeightAnalyzer
        import torch.nn as nn

        model = nn.Linear(10, 10)
        analyzer = WeightAnalyzer(model)

        stats = analyzer.compute_weight_statistics()

        assert "weight" in stats
        assert "mean" in stats["weight"]
        assert "std" in stats["weight"]


class TestVisualizeRegression:
    """可视化回归测试"""
    
    def test_attention_visualizer_creation(self):
        """测试注意力可视化器创建"""
        from model_probe.visualize import AttentionVisualizer
        
        viz = AttentionVisualizer()
        assert viz is not None
        
    def test_layer_visualizer_creation(self):
        """测试层可视化器创建"""
        from model_probe.visualize import LayerVisualizer
        
        viz = LayerVisualizer()
        assert viz is not None


class TestMemoryOptimizer:
    """显存优化回归测试"""

    def test_get_memory_usage(self):
        """测试显存使用获取"""
        from model_probe.core import MemoryOptimizer

        mem = MemoryOptimizer.get_memory_usage()

        assert "allocated" in mem
        assert "reserved" in mem

    def test_print_memory_usage(self):
        """测试显存打印（不报错）"""
        from model_probe.core import MemoryOptimizer

        MemoryOptimizer.print_memory_usage("Test: ")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

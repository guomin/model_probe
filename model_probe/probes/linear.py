"""
探针模块：用于理解模型内部表示
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class ProbeConfig:
    """探针配置"""
    hidden_dim: int = 768
    num_classes: int = 2
    probe_type: str = "linear"  # linear, mlp
    reg_strength: float = 1.0


class LinearProbe:
    """
    线性探针：用于探测模型隐藏层编码的信息
    训练一个线性分类器/回归器来预测标签，如果准确率高说明该层编码了这些信息
    """
    
    def __init__(self, config: Optional[ProbeConfig] = None):
        self.config = config or ProbeConfig()
        self.model: Optional[nn.Module] = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _build_model(self):
        """构建探针模型"""
        if self.config.probe_type == "linear":
            out_dim = 1 if self.config.num_classes == 2 else self.config.num_classes
            self.model = nn.Linear(self.config.hidden_dim, out_dim)
        elif self.config.probe_type == "mlp":
            self.model = nn.Sequential(
                nn.Linear(self.config.hidden_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, self.config.num_classes)
            )
            
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        val_split: float = 0.1,
        epochs: int = 100,
        lr: float = 0.001,
        batch_size: int = 64,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        训练探针
        
        Args:
            X: 隐藏层表示 (n_samples, hidden_dim)
            y: 标签 (n_samples,)
            val_split: 验证集比例
            epochs: 训练轮数
            lr: 学习率
            batch_size: 批次大小
            verbose: 是否打印训练过程
            
        Returns:
            训练历史
        """
        X = self.scaler.fit_transform(X)
        
        self._build_model()
        self.model.to(next(iter(torch.tensor([0]))).device)
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        n_train = int(len(X) * (1 - val_split))
        indices = torch.randperm(len(X))
        train_idx, val_idx = indices[:n_train], indices[n_train:]
        
        train_dataset = torch.utils.data.TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
        val_dataset = torch.utils.data.TensorDataset(X_tensor[val_idx], y_tensor[val_idx])
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
        
        if self.config.num_classes == 2:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
            
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        history = {"train_loss": [], "val_acc": []}
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                
                if self.config.num_classes == 2:
                    loss = criterion(outputs.squeeze(-1), batch_y.float())
                else:
                    loss = criterion(outputs, batch_y)
                    
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            train_loss /= len(train_loader)
            
            self.model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = self.model(batch_x)
                    _, predicted = torch.max(outputs, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                    
            val_acc = correct / total
            
            history["train_loss"].append(train_loss)
            history["val_acc"].append(val_acc)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")
                
        self.is_fitted = True
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("Probe not fitted yet")
            
        X = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            
        return predicted.numpy()
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算准确率"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> float:
        """交叉验证"""
        X_scaled = self.scaler.fit_transform(X)
        
        if self.config.probe_type == "linear":
            clf = LogisticRegression(C=self.config.reg_strength, max_iter=1000, solver='lbfgs', multi_class='multinomial')
        else:
            raise NotImplementedError("Cross-validation only supports linear probe")
            
        scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy")
        return scores.mean()


class RepresentationAnalyzer:
    """
    表示分析器：分析模型隐藏层的表示特性
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def compute_pca(
        self, 
        X: np.ndarray, 
        n_components: int = 50
    ) -> Tuple[np.ndarray, PCA]:
        """
        PCA降维分析
        
        Returns:
            降维后的表示, PCA对象
        """
        X_scaled = self.scaler.fit_transform(X)
        pca = PCA(n_components=min(n_components, X.shape[1]))
        X_pca = pca.fit_transform(X_scaled)
        return X_pca, pca
    
    def compute_tsne(
        self,
        X: np.ndarray,
        n_components: int = 2,
        perplexity: float = 30.0,
        n_iter: int = 1000
    ) -> np.ndarray:
        """
        t-SNE降维可视化
        """
        X_scaled = self.scaler.fit_transform(X)
        tsne = TSNE(
            n_components=n_components, 
            perplexity=perplexity, 
            n_iter=n_iter,
            random_state=42
        )
        return tsne.fit_transform(X_scaled)
    
    def compute_procrustes(
        self,
        X1: np.ndarray,
        X2: np.ndarray
    ) -> float:
        """
        计算两组表示的Procrustes距离（表示相似度）
        """
        from scipy.linalg import orthogonal_procrustes
        
        X1_scaled = self.scaler.fit_transform(X1)
        X2_scaled = self.scaler.fit_transform(X2)
        
        if X1_scaled.shape[1] != X2_scaled.shape[1]:
            min_dim = min(X1_scaled.shape[1], X2_scaled.shape[1])
            X1_scaled = X1_scaled[:, :min_dim]
            X2_scaled = X2_scaled[:, :min_dim]
            
        R, _ = orthogonal_procrustes(X2_scaled, X1_scaled)
        X2_aligned = X2_scaled @ R
        
        return np.sqrt(np.sum((X1_scaled - X2_aligned) ** 2))
    
    def compute_cosine_similarity_matrix(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """计算余弦相似度矩阵"""
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        return X_norm @ X_norm.T
    
    def analyze_similarity(
        self,
        representations: Dict[str, np.ndarray],
        method: str = "procrustes"
    ) -> Dict[Tuple[str, str], float]:
        """
        分析多层表示之间的相似度
        
        Args:
            representations: {layer_name: representations}
            method: procrustes 或 cosine
            
        Returns:
            相似度矩阵
        """
        layer_names = list(representations.keys())
        n_layers = len(layer_names)
        similarity_matrix = np.zeros((n_layers, n_layers))
        
        for i, layer1 in enumerate(layer_names):
            for j, layer2 in enumerate(layer_names):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                elif i < j:
                    if method == "procrustes":
                        sim = 1.0 / (1.0 + self.compute_procrustes(
                            representations[layer1], 
                            representations[layer2]
                        ))
                    else:
                        X1 = representations[layer1].mean(axis=0)
                        X2 = representations[layer2].mean(axis=0)
                        sim = np.dot(X1, X2) / (np.linalg.norm(X1) * np.linalg.norm(X2) + 1e-8)
                        
                    similarity_matrix[i, j] = similarity_matrix[j, i] = sim
                    
        return {(layer_names[i], layer_names[j]): similarity_matrix[i, j] 
                for i in range(n_layers) for j in range(n_layers)}
    
    def visualize_layers(
        self,
        representations: Dict[str, np.ndarray],
        output_path: str = "layer_similarity.png"
    ):
        """可视化层表示相似度"""
        layer_names = list(representations.keys())
        n_layers = len(layer_names)
        
        similarity_matrix = np.zeros((n_layers, n_layers))
        for i, layer1 in enumerate(layer_names):
            for j, layer2 in enumerate(layer_names):
                X1 = representations[layer1].mean(axis=0)
                X2 = representations[layer2].mean(axis=0)
                sim = np.dot(X1, X2) / (np.linalg.norm(X1) * np.linalg.norm(X2) + 1e-8)
                similarity_matrix[i, j] = sim
                
        plt.figure(figsize=(10, 8))
        plt.imshow(similarity_matrix, cmap="viridis")
        plt.colorbar(label="Cosine Similarity")
        plt.xticks(range(n_layers), layer_names, rotation=45, ha="right")
        plt.yticks(range(n_layers), layer_names)
        plt.title("Layer Representation Similarity")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"Saved similarity visualization to {output_path}")

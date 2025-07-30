# Social Pooling 測試規格文檔

## 概述

本文檔定義了 Social Pooling 模組的完整測試需求和規格，支援 TDD (Test-Driven Development) 開發流程。包括單元測試、整合測試、效能測試和邊界條件測試的詳細規格。

## 測試架構

### 測試層級

1. **單元測試**: 測試個別方法和函數的正確性
2. **整合測試**: 測試模組間的交互和資料流
3. **端到端測試**: 測試完整的空間聚合流程
4. **效能測試**: 測試計算效率和記憶體使用
5. **回歸測試**: 確保代碼變更不破壞現有功能

### 測試環境

```python
# 測試依賴套件
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import time
import gc

# 測試配置
TOLERANCE = 1e-6        # 數值比較容差
BATCH_SIZES = [1, 8, 32]  # 測試的批次大小
NUM_VDS_RANGE = [5, 20, 50, 100]  # 測試的 VD 數量範圍
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## 單元測試規格

### 1. SocialPoolingConfig 測試

#### 測試類別: `TestSocialPoolingConfig`

```python
class TestSocialPoolingConfig:
    """SocialPoolingConfig 類別的單元測試"""
    
    def test_default_initialization(self):
        """測試預設初始化"""
        # 期望行為
        config = SocialPoolingConfig()
        
        # 斷言檢查
        assert config.pooling_radius == 1000.0
        assert config.max_neighbors == 10
        assert config.distance_metric == "euclidean"
        assert config.weighting_function == "gaussian"
        assert config.enable_caching == True
        
    def test_custom_initialization(self):
        """測試自定義參數初始化"""
        config = SocialPoolingConfig(
            pooling_radius=500.0,
            max_neighbors=5,
            distance_metric="haversine",
            weighting_function="exponential"
        )
        
        assert config.pooling_radius == 500.0
        assert config.max_neighbors == 5
        assert config.distance_metric == "haversine"
        assert config.weighting_function == "exponential"
    
    def test_parameter_validation(self):
        """測試參數驗證"""
        # 無效的 pooling_radius
        with pytest.raises(ValueError, match="pooling_radius must be positive"):
            config = SocialPoolingConfig(pooling_radius=-100.0)
            config.validate()
        
        # 無效的 max_neighbors
        with pytest.raises(ValueError, match="max_neighbors must be >= min_neighbors"):
            config = SocialPoolingConfig(max_neighbors=2, min_neighbors=5)
            config.validate()
    
    def test_supported_distance_metrics(self):
        """測試支援的距離度量方法"""
        supported_metrics = ["euclidean", "manhattan", "haversine"]
        
        for metric in supported_metrics:
            config = SocialPoolingConfig(distance_metric=metric)
            assert config.distance_metric == metric
    
    def test_supported_weighting_functions(self):
        """測試支援的權重函數"""
        supported_functions = ["gaussian", "exponential", "inverse", "linear"]
        
        for func in supported_functions:
            config = SocialPoolingConfig(weighting_function=func)
            assert config.weighting_function == func
```

### 2. SocialPooling 核心方法測試

#### 測試類別: `TestSocialPooling`

```python
class TestSocialPooling:
    """SocialPooling 核心類別的單元測試"""
    
    @pytest.fixture
    def basic_config(self):
        """基本測試配置"""
        return SocialPoolingConfig(
            pooling_radius=1000.0,
            max_neighbors=5,
            weighting_function="gaussian",
            distance_metric="euclidean"
        )
    
    @pytest.fixture
    def sample_data(self):
        """樣本測試數據"""
        torch.manual_seed(42)
        return {
            'features': torch.randn(8, 12, 3),  # [batch, seq_len, features]
            'coordinates': torch.tensor([
                [0.0, 0.0],      # VD_000 at origin
                [100.0, 0.0],    # VD_001 100m east
                [0.0, 100.0],    # VD_002 100m north
                [100.0, 100.0],  # VD_003 northeast
                [500.0, 500.0]   # VD_004 far away
            ], dtype=torch.float32),
            'vd_ids': ['VD_000', 'VD_001', 'VD_002', 'VD_003', 'VD_004']
        }
    
    def test_initialization(self, basic_config):
        """測試 SocialPooling 初始化"""
        social_pooling = SocialPooling(basic_config)
        
        assert social_pooling.config == basic_config
        assert social_pooling._distance_cache is None
        assert social_pooling._coordinate_cache is None
    
    def test_calculate_distances_euclidean(self, basic_config, sample_data):
        """測試歐幾里得距離計算"""
        social_pooling = SocialPooling(basic_config)
        coordinates = sample_data['coordinates']
        
        distances = social_pooling.calculate_distances(coordinates)
        
        # 檢查形狀
        assert distances.shape == (5, 5)
        
        # 檢查對角線為零（自己到自己的距離）
        assert torch.allclose(torch.diag(distances), torch.zeros(5), atol=TOLERANCE)
        
        # 檢查對稱性
        assert torch.allclose(distances, distances.T, atol=TOLERANCE)
        
        # 檢查特定距離
        expected_distance_01 = 100.0  # VD_000 到 VD_001
        assert torch.allclose(distances[0, 1], torch.tensor(expected_distance_01), atol=TOLERANCE)
        
        expected_distance_03 = 141.42  # VD_000 到 VD_003 (對角線)
        assert torch.allclose(distances[0, 3], torch.tensor(expected_distance_03), atol=1.0)
    
    def test_calculate_distances_haversine(self, sample_data):
        """測試 Haversine 距離計算（地理座標）"""
        config = SocialPoolingConfig(distance_metric="haversine")
        social_pooling = SocialPooling(config)
        
        # 使用經緯度座標 (度)
        geo_coordinates = torch.tensor([
            [121.5, 25.0],    # 台北
            [121.6, 25.1],    # 台北附近
            [120.2, 23.0]     # 台南
        ], dtype=torch.float32)
        
        distances = social_pooling.calculate_distances(geo_coordinates)
        
        # 檢查基本屬性
        assert distances.shape == (3, 3)
        assert torch.allclose(torch.diag(distances), torch.zeros(3), atol=TOLERANCE)
        assert torch.allclose(distances, distances.T, atol=TOLERANCE)
        
        # 台北到台南的距離應該遠大於台北內部距離
        assert distances[0, 2] > distances[0, 1] * 10
    
    def test_compute_weights_gaussian(self, basic_config, sample_data):
        """測試高斯權重計算"""
        social_pooling = SocialPooling(basic_config)
        coordinates = sample_data['coordinates']
        
        distances = social_pooling.calculate_distances(coordinates)
        weights = social_pooling.compute_weights(distances)
        
        # 檢查形狀
        assert weights.shape == distances.shape
        
        # 檢查權重範圍 [0, 1]
        assert torch.all(weights >= 0)
        assert torch.all(weights <= 1)
        
        # 檢查對角線為 1（自己到自己的權重最大）
        assert torch.allclose(torch.diag(weights), torch.ones(5), atol=TOLERANCE)
        
        # 檢查距離遠的權重較小
        assert weights[0, 4] < weights[0, 1]  # 遠距離權重 < 近距離權重
    
    def test_compute_weights_exponential(self, sample_data):
        """測試指數衰減權重計算"""
        config = SocialPoolingConfig(weighting_function="exponential")
        social_pooling = SocialPooling(config)
        coordinates = sample_data['coordinates']
        
        distances = social_pooling.calculate_distances(coordinates)
        weights = social_pooling.compute_weights(distances)
        
        # 檢查基本屬性
        assert weights.shape == distances.shape
        assert torch.all(weights >= 0)
        assert torch.all(weights <= 1)
        assert torch.allclose(torch.diag(weights), torch.ones(5), atol=TOLERANCE)
    
    def test_find_neighbors(self, basic_config, sample_data):
        """測試鄰居發現"""
        social_pooling = SocialPooling(basic_config)
        coordinates = sample_data['coordinates']
        
        distances = social_pooling.calculate_distances(coordinates)
        neighbor_indices, neighbor_distances = social_pooling.find_neighbors(distances)
        
        # 檢查形狀
        max_neighbors = basic_config.max_neighbors
        assert neighbor_indices.shape == (5, max_neighbors)
        assert neighbor_distances.shape == (5, max_neighbors)
        
        # 檢查每個節點的第一個鄰居是自己
        assert torch.allclose(neighbor_indices[:, 0], torch.arange(5).float())
        assert torch.allclose(neighbor_distances[:, 0], torch.zeros(5))
        
        # 檢查鄰居距離遞增
        for i in range(5):
            neighbors_dist = neighbor_distances[i]
            valid_neighbors = neighbors_dist[neighbors_dist < float('inf')]
            if len(valid_neighbors) > 1:
                assert torch.all(valid_neighbors[1:] >= valid_neighbors[:-1])
    
    def test_aggregate_features(self, basic_config, sample_data):
        """測試特徵聚合"""
        social_pooling = SocialPooling(basic_config)
        features = sample_data['features']
        coordinates = sample_data['coordinates']
        
        distances = social_pooling.calculate_distances(coordinates)
        weights = social_pooling.compute_weights(distances)
        neighbor_indices, _ = social_pooling.find_neighbors(distances)
        
        # 模擬聚合（簡化測試）
        aggregated = social_pooling.aggregate_features(features, weights, neighbor_indices)
        
        # 檢查形狀保持不變
        assert aggregated.shape == features.shape
        
        # 檢查數值變化（聚合後應該有所不同）
        assert not torch.allclose(aggregated, features, atol=TOLERANCE)
    
    def test_forward_pass(self, basic_config, sample_data):
        """測試完整前向傳播"""
        social_pooling = SocialPooling(basic_config)
        
        features = sample_data['features']
        coordinates = sample_data['coordinates']
        vd_ids = sample_data['vd_ids']
        
        # 執行前向傳播
        output = social_pooling(features, coordinates, vd_ids)
        
        # 檢查輸出形狀
        assert output.shape == features.shape
        
        # 檢查輸出類型
        assert isinstance(output, torch.Tensor)
        assert output.dtype == features.dtype
        
        # 檢查梯度可計算性
        if features.requires_grad:
            assert output.requires_grad
```

### 3. 邊界條件測試

#### 測試類別: `TestSocialPoolingEdgeCases`

```python
class TestSocialPoolingEdgeCases:
    """邊界條件和異常情況測試"""
    
    def test_single_vd(self):
        """測試單個 VD 的情況"""
        config = SocialPoolingConfig()
        social_pooling = SocialPooling(config)
        
        features = torch.randn(1, 12, 3)
        coordinates = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        vd_ids = ['VD_000']
        
        output = social_pooling(features, coordinates, vd_ids)
        
        # 單個 VD 時，輸出應該等於輸入（無鄰居聚合）
        assert torch.allclose(output, features, atol=TOLERANCE)
    
    def test_two_vds_close(self):
        """測試兩個靠近的 VD"""
        config = SocialPoolingConfig(pooling_radius=500.0, max_neighbors=2)
        social_pooling = SocialPooling(config)
        
        features = torch.randn(2, 12, 3)
        coordinates = torch.tensor([[0.0, 0.0], [100.0, 0.0]], dtype=torch.float32)
        vd_ids = ['VD_000', 'VD_001']
        
        output = social_pooling(features, coordinates, vd_ids)
        
        # 兩個靠近的 VD 應該互相影響
        assert output.shape == features.shape
        assert not torch.allclose(output, features, atol=TOLERANCE)
    
    def test_two_vds_far(self):
        """測試兩個遠距離的 VD"""
        config = SocialPoolingConfig(pooling_radius=500.0, max_neighbors=2)
        social_pooling = SocialPooling(config)
        
        features = torch.randn(2, 12, 3)
        coordinates = torch.tensor([[0.0, 0.0], [2000.0, 0.0]], dtype=torch.float32)  # 超出半徑
        vd_ids = ['VD_000', 'VD_001']
        
        output = social_pooling(features, coordinates, vd_ids)
        
        # 遠距離的 VD 不應該互相影響
        assert output.shape == features.shape
        # 輸出可能接近原始輸入（取決於具體實現）
    
    def test_empty_coordinates(self):
        """測試空座標情況"""
        config = SocialPoolingConfig()
        social_pooling = SocialPooling(config)
        
        with pytest.raises(RuntimeError):
            features = torch.randn(0, 12, 3)
            coordinates = torch.empty((0, 2), dtype=torch.float32)
            vd_ids = []
            social_pooling(features, coordinates, vd_ids)
    
    def test_mismatched_dimensions(self):
        """測試維度不匹配情況"""
        config = SocialPoolingConfig()
        social_pooling = SocialPooling(config)
        
        features = torch.randn(5, 12, 3)
        coordinates = torch.randn(3, 2)  # 數量不匹配
        vd_ids = ['VD_000', 'VD_001', 'VD_002']
        
        with pytest.raises(ValueError):
            social_pooling(features, coordinates, vd_ids)
    
    def test_invalid_coordinates(self):
        """測試無效座標（NaN, Inf）"""
        config = SocialPoolingConfig()
        social_pooling = SocialPooling(config)
        
        features = torch.randn(2, 12, 3)
        coordinates = torch.tensor([[0.0, float('nan')], [1.0, float('inf')]], dtype=torch.float32)
        vd_ids = ['VD_000', 'VD_001']
        
        with pytest.raises(ValueError):
            social_pooling(features, coordinates, vd_ids)
    
    def test_zero_radius(self):
        """測試零半徑情況"""
        config = SocialPoolingConfig(pooling_radius=0.0)
        
        with pytest.raises(ValueError):
            config.validate()
    
    def test_max_neighbors_zero(self):
        """測試最大鄰居數為零"""
        config = SocialPoolingConfig(max_neighbors=0)
        social_pooling = SocialPooling(config)
        
        features = torch.randn(3, 12, 3)
        coordinates = torch.randn(3, 2)
        vd_ids = ['VD_000', 'VD_001', 'VD_002']
        
        output = social_pooling(features, coordinates, vd_ids)
        
        # 無鄰居時，輸出應該是零或保持原樣（取決於實現）
        assert output.shape == features.shape
```

## 整合測試規格

### 測試類別: `TestSocialPoolingIntegration`

```python
class TestSocialPoolingIntegration:
    """Social Pooling 與其他模組的整合測試"""
    
    def test_integration_with_traffic_lstm(self):
        """測試與 TrafficLSTM 的整合"""
        from social_xlstm.models.lstm import TrafficLSTM, TrafficLSTMConfig
        
        # 配置
        lstm_config = TrafficLSTMConfig(hidden_size=32, num_layers=1)
        social_config = SocialPoolingConfig(pooling_radius=1000.0)
        
        # 創建模型
        lstm_model = TrafficLSTM(lstm_config)
        social_pooling = SocialPooling(social_config)
        
        # 測試數據
        features = torch.randn(4, 12, 3)
        coordinates = torch.randn(10, 2) * 1000
        vd_ids = [f"VD_{i:03d}" for i in range(10)]
        
        # Post-Fusion 策略測試
        social_features = social_pooling(features, coordinates, vd_ids)
        lstm_output = lstm_model(social_features)
        
        assert lstm_output.shape == (4, 1, 3)  # 預期輸出形狀
        assert isinstance(lstm_output, torch.Tensor)
    
    def test_integration_with_coordinate_system(self):
        """測試與座標系統的整合"""
        from social_xlstm.utils.spatial_coords import CoordinateSystem
        
        config = SocialPoolingConfig(distance_metric="haversine")
        social_pooling = SocialPooling(config)
        coord_system = CoordinateSystem()
        
        # 使用真實座標轉換
        latlon_coords = torch.tensor([[25.0, 121.5], [25.1, 121.6]], dtype=torch.float32)
        xy_coords = coord_system.latlon_to_xy(latlon_coords)
        
        features = torch.randn(2, 12, 3)
        vd_ids = ['VD_001', 'VD_002']
        
        # 測試兩種座標系統
        output_geo = social_pooling(features, latlon_coords, vd_ids)
        output_xy = social_pooling(features, xy_coords, vd_ids)
        
        assert output_geo.shape == features.shape
        assert output_xy.shape == features.shape
        # 不同座標系統可能產生不同結果
    
    def test_factory_function_integration(self):
        """測試工廠函數整合"""
        from social_xlstm.models.social_pooling import create_social_traffic_model
        from social_xlstm.models.lstm import TrafficLSTMConfig
        
        lstm_config = TrafficLSTMConfig(hidden_size=32)
        social_config = SocialPoolingConfig()
        
        # 測試 Post-Fusion 策略
        model_post = create_social_traffic_model(
            base_model_type="lstm",
            strategy="post_fusion",
            base_config=lstm_config,
            social_config=social_config
        )
        
        # 測試 Internal Gate Injection 策略
        model_igi = create_social_traffic_model(
            base_model_type="lstm",
            strategy="internal_injection",
            base_config=lstm_config,
            social_config=social_config
        )
        
        # 測試數據
        features = torch.randn(2, 12, 3)
        coordinates = torch.randn(5, 2)
        vd_ids = [f"VD_{i}" for i in range(5)]
        
        output_post = model_post(features, coordinates, vd_ids)
        output_igi = model_igi(features, coordinates, vd_ids)
        
        assert output_post.shape == (2, 1, 3)
        assert output_igi.shape == (2, 1, 3)
        assert not torch.allclose(output_post, output_igi, atol=TOLERANCE)  # 不同策略應產生不同結果
```

## 效能測試規格

### 測試類別: `TestSocialPoolingPerformance`

```python
class TestSocialPoolingPerformance:
    """效能和擴展性測試"""
    
    @pytest.mark.parametrize("num_vds", [10, 50, 100, 200])
    def test_scalability_with_vd_count(self, num_vds):
        """測試 VD 數量擴展性"""
        config = SocialPoolingConfig(max_neighbors=min(10, num_vds))
        social_pooling = SocialPooling(config)
        
        features = torch.randn(8, 12, 3)
        coordinates = torch.randn(num_vds, 2) * 2000
        vd_ids = [f"VD_{i:03d}" for i in range(num_vds)]
        
        # 計時測試
        start_time = time.time()
        for _ in range(10):
            _ = social_pooling(features, coordinates, vd_ids)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        
        # 效能基準：每個操作應在合理時間內完成
        if num_vds <= 50:
            assert avg_time < 0.1  # 100ms
        elif num_vds <= 100:
            assert avg_time < 0.5  # 500ms
        else:
            assert avg_time < 2.0  # 2s
    
    @pytest.mark.parametrize("batch_size", [1, 8, 32, 64])
    def test_scalability_with_batch_size(self, batch_size):
        """測試批次大小擴展性"""
        config = SocialPoolingConfig()
        social_pooling = SocialPooling(config)
        
        features = torch.randn(batch_size, 12, 3)
        coordinates = torch.randn(20, 2) * 1000
        vd_ids = [f"VD_{i:03d}" for i in range(20)]
        
        # 記憶體使用測試
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated()
        
        output = social_pooling(features, coordinates, vd_ids)
        
        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated()
            memory_used = memory_after - memory_before
            
            # 記憶體使用應該合理
            expected_memory = batch_size * 12 * 3 * 4 * 2  # 大致估算
            assert memory_used < expected_memory * 10  # 允許 10 倍的緩衝
        
        assert output.shape == features.shape
    
    def test_caching_performance(self):
        """測試快取效能提升"""
        config_cached = SocialPoolingConfig(enable_caching=True)
        config_no_cache = SocialPoolingConfig(enable_caching=False)
        
        social_pooling_cached = SocialPooling(config_cached)
        social_pooling_no_cache = SocialPooling(config_no_cache)
        
        features = torch.randn(16, 12, 3)
        coordinates = torch.randn(50, 2) * 1000
        vd_ids = [f"VD_{i:03d}" for i in range(50)]
        
        # 測試快取版本（第二次呼叫應該更快）
        start_time = time.time()
        _ = social_pooling_cached(features, coordinates, vd_ids)
        first_call_time = time.time() - start_time
        
        start_time = time.time()
        _ = social_pooling_cached(features, coordinates, vd_ids)  # 使用快取
        second_call_time = time.time() - start_time
        
        # 測試無快取版本
        start_time = time.time()
        _ = social_pooling_no_cache(features, coordinates, vd_ids)
        no_cache_time = time.time() - start_time
        
        # 快取應該提升第二次呼叫的效能
        assert second_call_time <= first_call_time * 0.8  # 至少 20% 提升
    
    def test_memory_efficiency(self):
        """測試記憶體效率"""
        config = SocialPoolingConfig()
        social_pooling = SocialPooling(config)
        
        # 大數據測試
        large_features = torch.randn(64, 24, 3)
        large_coordinates = torch.randn(100, 2) * 5000
        large_vd_ids = [f"VD_{i:04d}" for i in range(100)]
        
        if torch.cuda.is_available():
            large_features = large_features.cuda()
            large_coordinates = large_coordinates.cuda()
            social_pooling = social_pooling.cuda()
            
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated()
        
        # 執行多次以檢查記憶體洩漏
        for _ in range(5):
            output = social_pooling(large_features, large_coordinates, large_vd_ids)
            del output
            gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_after = torch.cuda.memory_allocated()
            
            # 記憶體使用不應顯著增加（檢查洩漏）
            memory_increase = memory_after - memory_before
            assert memory_increase < 1024 * 1024 * 100  # 少於 100MB 增加
```

## 數值穩定性測試

### 測試類別: `TestSocialPoolingNumericalStability`

```python
class TestSocialPoolingNumericalStability:
    """數值穩定性和精度測試"""
    
    def test_distance_calculation_precision(self):
        """測試距離計算精度"""
        config = SocialPoolingConfig(distance_metric="euclidean")
        social_pooling = SocialPooling(config)
        
        # 高精度測試數據
        coordinates = torch.tensor([
            [0.0, 0.0],
            [1e-6, 0.0],      # 很小的距離
            [1e6, 0.0],       # 很大的距離
            [1.23456789, 4.56789123]  # 高精度座標
        ], dtype=torch.float64)  # 使用雙精度
        
        distances = social_pooling.calculate_distances(coordinates)
        
        # 檢查小距離精度
        small_distance = distances[0, 1].item()
        expected_small = 1e-6
        assert abs(small_distance - expected_small) < 1e-12
        
        # 檢查大距離精度
        large_distance = distances[0, 2].item()
        expected_large = 1e6
        assert abs(large_distance - expected_large) < 1e-6
    
    def test_weight_function_stability(self):
        """測試權重函數的數值穩定性"""
        config = SocialPoolingConfig(weighting_function="gaussian")
        social_pooling = SocialPooling(config)
        
        # 極端距離值測試
        extreme_distances = torch.tensor([
            [0.0, 1e-10, 1e10],
            [1e-10, 0.0, 1e10],
            [1e10, 1e10, 0.0]
        ], dtype=torch.float32)
        
        weights = social_pooling.compute_weights(extreme_distances)
        
        # 檢查權重範圍
        assert torch.all(weights >= 0)
        assert torch.all(weights <= 1)
        assert not torch.any(torch.isnan(weights))
        assert not torch.any(torch.isinf(weights))
        
        # 對角線應該是 1
        assert torch.allclose(torch.diag(weights), torch.ones(3), atol=TOLERANCE)
    
    def test_gradient_flow(self):
        """測試梯度流的穩定性"""
        config = SocialPoolingConfig()
        social_pooling = SocialPooling(config)
        
        # 啟用梯度追蹤
        features = torch.randn(4, 10, 3, requires_grad=True)
        coordinates = torch.randn(8, 2, requires_grad=True)
        vd_ids = [f"VD_{i}" for i in range(8)]
        
        output = social_pooling(features, coordinates, vd_ids)
        loss = output.sum()
        loss.backward()
        
        # 檢查梯度存在且有限
        assert features.grad is not None
        assert coordinates.grad is not None
        assert torch.all(torch.isfinite(features.grad))
        assert torch.all(torch.isfinite(coordinates.grad))
        
        # 檢查梯度不為零（確保有學習信號）
        assert torch.any(features.grad != 0)
        assert torch.any(coordinates.grad != 0)
```

## 測試數據生成器

### 合成數據生成

```python
class SyntheticDataGenerator:
    """合成測試數據生成器"""
    
    @staticmethod
    def generate_grid_coordinates(grid_size: int, spacing: float = 1000.0) -> torch.Tensor:
        """生成網格狀的 VD 座標"""
        x = torch.arange(grid_size, dtype=torch.float32) * spacing
        y = torch.arange(grid_size, dtype=torch.float32) * spacing
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        coordinates = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        return coordinates
    
    @staticmethod
    def generate_traffic_features(batch_size: int, seq_len: int, num_features: int = 3) -> torch.Tensor:
        """生成具有真實交通特性的特徵數據"""
        # 基本交通模式：白天高峰、夜間低峰
        time_pattern = torch.sin(torch.linspace(0, 4*torch.pi, seq_len)) * 0.3 + 0.7
        
        features = torch.randn(batch_size, seq_len, num_features)
        
        # 應用時間模式到體積特徵
        features[:, :, 0] *= time_pattern.unsqueeze(0)  # 車流量
        features[:, :, 1] = torch.clamp(features[:, :, 1], 0, 120)  # 速度限制 0-120 km/h
        features[:, :, 2] = torch.sigmoid(features[:, :, 2])  # 佔有率 0-1
        
        return features
    
    @staticmethod
    def generate_realistic_coordinates(num_vds: int, area_km: float = 10.0) -> torch.Tensor:
        """生成具有真實道路網絡特性的座標"""
        # 使用 Poisson 點過程生成非均勻分布
        base_coords = torch.rand(num_vds, 2) * area_km * 1000  # 轉換為公尺
        
        # 添加聚類效應（模擬城市道路聚集）
        cluster_centers = torch.rand(max(1, num_vds // 10), 2) * area_km * 1000
        
        for i in range(num_vds):
            nearest_cluster = torch.argmin(torch.norm(base_coords[i:i+1] - cluster_centers, dim=1))
            cluster_center = cluster_centers[nearest_cluster]
            # 向聚類中心偏移
            direction = cluster_center - base_coords[i]
            base_coords[i] += direction * 0.3 * torch.rand(1)
        
        return base_coords
```

## 測試執行配置

### pytest 配置

```python
# pytest.ini 配置建議
[tool:pytest]
testpaths = tests/
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --tb=short
    --strict-markers
    --disable-warnings
markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
    slow: Slow running tests
    gpu: Tests requiring GPU
```

### 測試執行命令

```bash
# 執行所有測試
pytest tests/test_social_pooling/

# 執行特定類型的測試
pytest -m unit tests/test_social_pooling/
pytest -m integration tests/test_social_pooling/
pytest -m performance tests/test_social_pooling/

# 執行效能測試（較慢）
pytest -m slow tests/test_social_pooling/

# 產生覆蓋率報告
pytest --cov=social_xlstm.models.social_pooling tests/test_social_pooling/

# 平行執行測試
pytest -n auto tests/test_social_pooling/
```

## 持續整合配置

### GitHub Actions 工作流程

```yaml
name: Social Pooling Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
        torch-version: [1.13.0, 2.0.0]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install torch==${{ matrix.torch-version }}
        pip install -e .
        pip install pytest pytest-cov pytest-xdist
    
    - name: Run unit tests
      run: pytest -m unit tests/test_social_pooling/
    
    - name: Run integration tests
      run: pytest -m integration tests/test_social_pooling/
    
    - name: Generate coverage report
      run: pytest --cov=social_xlstm.models.social_pooling tests/test_social_pooling/
```

## 測試報告模板

### 測試結果報告

```markdown
# Social Pooling 測試報告

## 測試概要
- **執行時間**: {datetime}
- **測試環境**: Python {version}, PyTorch {torch_version}
- **總測試數**: {total_tests}
- **通過**: {passed_tests}
- **失敗**: {failed_tests}
- **跳過**: {skipped_tests}

## 覆蓋率報告
- **整體覆蓋率**: {overall_coverage}%
- **SocialPooling 類別**: {class_coverage}%
- **配置模組**: {config_coverage}%
- **工廠函數**: {factory_coverage}%

## 效能基準
- **平均執行時間**: {avg_execution_time}ms
- **記憶體使用峰值**: {peak_memory}MB
- **擴展性測試**: {scalability_results}

## 已知問題
{known_issues_list}

## 建議
{recommendations_list}
```

## 相關資源

- [Social Pooling API 參考](../implementation/social_pooling_api.md)
- [開發工作流程指南](../guides/social_pooling_development.md)
- [整合測試指南](../guides/social_pooling_testing_guide.md)
- [效能最佳化指南](../technical/social_pooling_optimization.md)
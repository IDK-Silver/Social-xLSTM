# Social-xLSTM 測試套件

## 📋 測試結構

```
tests/
├── conftest.py                 # 全局 pytest 配置和 fixtures
├── pytest.ini                 # pytest 配置文件
├── README.md                   # 本文件
│
├── unit/                       # 單元測試
│   ├── conftest.py            # 單元測試專用 fixtures
│   ├── dataset/               # 數據集模組測試
│   │   ├── test_config.py
│   │   ├── test_extract_archive.py
│   │   ├── test_h5_operations.py
│   │   ├── test_integration.py
│   │   ├── test_json_utils.py
│   │   ├── test_processor.py
│   │   └── test_zip_utils.py
│   ├── models/                # 模型相關測試
│   │   └── (待添加)
│   ├── training/              # 訓練相關測試
│   │   └── test_recorder.py
│   ├── visualization/         # 視覺化測試
│   │   └── test_training_visualizer.py
│   └── utils/                 # 工具函數測試
│       └── (待添加)
│
├── integration/               # 整合測試
│   ├── conftest.py           # 整合測試專用 fixtures
│   └── test_training_pipeline.py
│
├── functional/               # 功能測試
│   └── test_end_to_end.py
│
└── fixtures/                 # 測試數據和 fixtures
    ├── __init__.py
    └── sample_data.py
```

## 🎯 測試類型

### 1. 單元測試 (Unit Tests)
- **目的**：測試個別模組和函數
- **特點**：快速、獨立、專注於單一功能
- **標記**：`@pytest.mark.unit`

### 2. 整合測試 (Integration Tests)
- **目的**：測試模組間的互動
- **特點**：測試多個組件協同工作
- **標記**：`@pytest.mark.integration`

### 3. 功能測試 (Functional Tests)
- **目的**：測試完整的用戶工作流程
- **特點**：端到端測試，從用戶角度驗證
- **標記**：`@pytest.mark.functional`

## 🚀 運行測試

### 基本命令

```bash
# 運行所有測試
pytest

# 運行特定類型的測試
pytest -m unit              # 只運行單元測試
pytest -m integration       # 只運行整合測試
pytest -m functional        # 只運行功能測試

# 運行特定目錄的測試
pytest tests/unit/           # 只運行單元測試
pytest tests/integration/   # 只運行整合測試

# 運行特定文件
pytest tests/unit/training/test_recorder.py

# 運行特定測試函數
pytest tests/unit/training/test_recorder.py::TestTrainingRecorder::test_initialization
```

### 進階選項

```bash
# 顯示詳細輸出
pytest -v

# 並行運行測試（需要 pytest-xdist）
pytest -n auto

# 只運行失敗的測試
pytest --lf

# 停止在第一個失敗
pytest -x

# 顯示最慢的 10 個測試
pytest --durations=10

# 跳過慢速測試
pytest -m "not slow"

# 只運行快速測試
pytest -m "not slow and not gpu"
```

### 測試覆蓋率

```bash
# 生成覆蓋率報告（需要 pytest-cov）
pytest --cov=src --cov-report=html
pytest --cov=src --cov-report=term-missing
```

## 📊 測試標記

### 內建標記
- `unit`: 單元測試
- `integration`: 整合測試  
- `functional`: 功能測試
- `slow`: 慢速測試（可用 `-m "not slow"` 跳過）
- `gpu`: 需要 GPU 的測試

### 使用標記

```python
import pytest

@pytest.mark.unit
def test_simple_function():
    pass

@pytest.mark.slow
@pytest.mark.gpu
def test_gpu_training():
    pass

@pytest.mark.integration
def test_data_pipeline():
    pass
```

## 🔧 測試配置

### 全局配置 (conftest.py)
- 共用的 fixtures
- 測試環境設置
- 隨機種子設定
- 臨時目錄管理

### 測試專用配置
- `unit/conftest.py`: 單元測試專用 fixtures
- `integration/conftest.py`: 整合測試專用 fixtures

## 📝 編寫測試

### 測試命名規範
- 測試文件：`test_*.py`
- 測試類：`Test*`
- 測試函數：`test_*`

### 基本測試結構

```python
import pytest
from social_xlstm.some_module import SomeClass

class TestSomeClass:
    """測試 SomeClass 的功能"""
    
    @pytest.fixture
    def sample_instance(self):
        """創建測試實例"""
        return SomeClass()
    
    def test_basic_functionality(self, sample_instance):
        """測試基本功能"""
        result = sample_instance.do_something()
        assert result == expected_value
    
    def test_error_handling(self, sample_instance):
        """測試錯誤處理"""
        with pytest.raises(ValueError):
            sample_instance.do_something_invalid()
```

### 使用 Fixtures

```python
def test_with_temp_dir(temp_dir):
    """使用臨時目錄的測試"""
    test_file = temp_dir / "test.txt"
    test_file.write_text("test content")
    assert test_file.exists()

def test_with_sample_data(sample_data):
    """使用樣本數據的測試"""
    assert sample_data['input'].shape == (16, 12, 3)
```

## 🐛 調試測試

### 常用調試技巧

```bash
# 進入 pdb 調試器
pytest --pdb

# 在第一個失敗時進入 pdb
pytest --pdb -x

# 顯示本地變量
pytest --tb=long

# 不捕獲輸出（顯示 print 語句）
pytest -s
```

### 測試中的 print 調試

```python
def test_with_debug_output(sample_data):
    print(f"Input shape: {sample_data['input'].shape}")
    print(f"Target shape: {sample_data['target'].shape}")
    
    # 測試邏輯...
    assert True
```

## 📈 測試最佳實踐

### 1. 測試獨立性
- 每個測試應該能獨立運行
- 不依賴其他測試的結果
- 使用 fixtures 設置測試環境

### 2. 測試數據管理
- 使用 fixtures 生成測試數據
- 避免硬編碼測試數據
- 使用臨時目錄處理文件

### 3. 錯誤處理測試
- 測試正常情況和錯誤情況
- 使用 `pytest.raises` 測試異常
- 測試邊界條件

### 4. 性能考慮
- 標記慢速測試
- 使用適當的測試數據大小
- 考慮並行執行

## 🔄 持續集成

### GitHub Actions 配置示例

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## 📚 相關文檔

- [pytest 官方文檔](https://docs.pytest.org/)
- [pytest fixtures 文檔](https://docs.pytest.org/en/stable/fixture.html)
- [pytest-cov 文檔](https://pytest-cov.readthedocs.io/)
- [pytest-xdist 文檔](https://pytest-xdist.readthedocs.io/)

## 🆘 常見問題

### Q: 測試運行很慢怎麼辦？
A: 
- 使用 `-m "not slow"` 跳過慢速測試
- 使用 `-n auto` 並行運行
- 檢查是否有不必要的重複設置

### Q: 如何跳過某些測試？
A:
```python
@pytest.mark.skip(reason="Not implemented yet")
def test_future_feature():
    pass

@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
def test_gpu_feature():
    pass
```

### Q: 如何設置測試環境變數？
A:
```python
import os
import pytest

@pytest.fixture(autouse=True)
def setup_env():
    os.environ['TEST_MODE'] = 'true'
    yield
    del os.environ['TEST_MODE']
```
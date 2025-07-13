# Snakefile xLSTM 整合設計

**日期**: 2025-07-13  
**依據**: ADR-0501 xLSTM 整合策略  
**目標**: 設計並行 LSTM/xLSTM 訓練工作流程  

## 📋 設計目標

### 核心需求
1. **並行訓練**: LSTM 和 xLSTM 可同時訓練，無需切換配置文件
2. **配置分離**: 各自有專門的配置區塊，參數互不干擾
3. **結果比較**: 統一的輸出格式，便於性能比較
4. **向後兼容**: 現有 LSTM 工作流程保持不變

### 技術要求
- 支持三種訓練模式：Single VD, Multi-VD, Independent Multi-VD
- 每種模式都有 LSTM 和 xLSTM 版本（共6個訓練規則）
- 統一的報告和視覺化生成
- 自動化的性能比較規則

## 🏗️ 配置結構設計

### 當前配置結構（現有）
```yaml
# cfgs/snakemake/default.yaml
training:
  single_vd:
    model_type: "lstm"
    experiment_dir: "blob/experiments/dev/single_vd"
    epochs: 100
    batch_size: 32
    # ... 其他 LSTM 參數
```

### 新增 xLSTM 配置結構（提議）
```yaml
# cfgs/snakemake/default.yaml  
training:
  # 現有 LSTM 配置保持不變
  single_vd:
    model_type: "lstm"
    experiment_dir: "blob/experiments/dev/single_vd_lstm"
    epochs: 100
    batch_size: 32
    sequence_length: 20
    # LSTM 特有參數
    hidden_size: 256
    num_layers: 3
    dropout: 0.3
    
  multi_vd:
    model_type: "lstm"
    experiment_dir: "blob/experiments/dev/multi_vd_lstm"
    # ... LSTM Multi-VD 參數
    
  independent_multi_vd:
    model_type: "lstm" 
    experiment_dir: "blob/experiments/dev/independent_multi_vd_lstm"
    # ... LSTM Independent Multi-VD 參數

# 新增 xLSTM 配置區塊
training_xlstm:
  single_vd:
    model_type: "xlstm"
    experiment_dir: "blob/experiments/dev/single_vd_xlstm"
    epochs: 100
    batch_size: 32
    sequence_length: 20
    # xLSTM 特有參數
    embedding_dim: 128
    num_blocks: 6
    slstm_at: [1, 3]
    slstm_backend: "vanilla"
    mlstm_backend: "vanilla"
    dropout: 0.1
    context_length: 256
    
  multi_vd:
    model_type: "xlstm"
    experiment_dir: "blob/experiments/dev/multi_vd_xlstm"
    epochs: 100
    batch_size: 16  # 可能需要較小的 batch size
    sequence_length: 20
    # xLSTM Multi-VD 特有參數
    embedding_dim: 128
    num_blocks: 6
    slstm_at: [1, 3, 5]
    num_vds: 10
    dropout: 0.1
    
  independent_multi_vd:
    model_type: "xlstm"
    experiment_dir: "blob/experiments/dev/independent_multi_vd_xlstm"
    epochs: 100
    batch_size: 16
    sequence_length: 20
    # xLSTM Independent Multi-VD 參數
    embedding_dim: 128
    num_blocks: 6
    slstm_at: [1, 3]
    num_vds: 10
    target_vd_index: 0
    dropout: 0.1
```

## 🔧 Snakefile 規則設計

### 新增 xLSTM 訓練規則

```python
# Snakefile 新增規則

# xLSTM Single VD 訓練
rule train_single_vd_xlstm:
    input:
        h5_file=config['dataset']['pre-processed']['h5']['file']
    output:
        model_file=os.path.join(config['training_xlstm']['single_vd']['experiment_dir'], "best_model.pt"),
        config_file=os.path.join(config['training_xlstm']['single_vd']['experiment_dir'], "config.json"),
        training_history=os.path.join(config['training_xlstm']['single_vd']['experiment_dir'], "training_history.json")
    log:
        config['training_xlstm']['single_vd']['log']
    params:
        epochs=config['training_xlstm']['single_vd']['epochs'],
        batch_size=config['training_xlstm']['single_vd']['batch_size'],
        sequence_length=config['training_xlstm']['single_vd']['sequence_length'],
        model_type=config['training_xlstm']['single_vd']['model_type'],
        experiment_name=os.path.basename(config['training_xlstm']['single_vd']['experiment_dir']),
        embedding_dim=config['training_xlstm']['single_vd']['embedding_dim'],
        num_blocks=config['training_xlstm']['single_vd']['num_blocks'],
        slstm_at=config['training_xlstm']['single_vd']['slstm_at']
    shell:
        """
        python scripts/train/without_social_pooling/train_single_vd.py \
        --data_path {input.h5_file} \
        --epochs {params.epochs} \
        --batch_size {params.batch_size} \
        --sequence_length {params.sequence_length} \
        --model_type {params.model_type} \
        --experiment_name {params.experiment_name} \
        --save_dir $(dirname $(dirname {output.model_file})) \
        --embedding_dim {params.embedding_dim} \
        --num_blocks {params.num_blocks} \
        --slstm_at {params.slstm_at} >> {log} 2>&1
        """

# xLSTM Multi-VD 訓練
rule train_multi_vd_xlstm:
    input:
        h5_file=config['dataset']['pre-processed']['h5']['file']
    output:
        training_history=os.path.join(config['training_xlstm']['multi_vd']['experiment_dir'], "training_history.json"),
        model_file=os.path.join(config['training_xlstm']['multi_vd']['experiment_dir'], "best_model.pth")
    log:
        config['training_xlstm']['multi_vd']['log']
    params:
        epochs=config['training_xlstm']['multi_vd']['epochs'],
        batch_size=config['training_xlstm']['multi_vd']['batch_size'],
        sequence_length=config['training_xlstm']['multi_vd']['sequence_length'],
        num_vds=config['training_xlstm']['multi_vd']['num_vds'],
        model_type=config['training_xlstm']['multi_vd']['model_type'],
        experiment_name=os.path.basename(config['training_xlstm']['multi_vd']['experiment_dir']),
        embedding_dim=config['training_xlstm']['multi_vd']['embedding_dim'],
        num_blocks=config['training_xlstm']['multi_vd']['num_blocks'],
        slstm_at=config['training_xlstm']['multi_vd']['slstm_at']
    shell:
        """
        python scripts/train/without_social_pooling/train_multi_vd.py \
        --data_path {input.h5_file} \
        --epochs {params.epochs} \
        --batch_size {params.batch_size} \
        --sequence_length {params.sequence_length} \
        --num_vds {params.num_vds} \
        --model_type {params.model_type} \
        --experiment_name {params.experiment_name} \
        --save_dir $(dirname {output.training_history}) \
        --embedding_dim {params.embedding_dim} \
        --num_blocks {params.num_blocks} \
        --slstm_at {params.slstm_at} >> {log} 2>&1
        """

# xLSTM Independent Multi-VD 訓練  
rule train_independent_multi_vd_xlstm:
    input:
        h5_file=config['dataset']['pre-processed']['h5']['file']
    output:
        training_history=os.path.join(config['training_xlstm']['independent_multi_vd']['experiment_dir'], "training_history.json"),
        model_file=os.path.join(config['training_xlstm']['independent_multi_vd']['experiment_dir'], "best_model.pth")
    log:
        config['training_xlstm']['independent_multi_vd']['log']
    params:
        epochs=config['training_xlstm']['independent_multi_vd']['epochs'],
        batch_size=config['training_xlstm']['independent_multi_vd']['batch_size'],
        sequence_length=config['training_xlstm']['independent_multi_vd']['sequence_length'],
        num_vds=config['training_xlstm']['independent_multi_vd']['num_vds'],
        target_vd_index=config['training_xlstm']['independent_multi_vd']['target_vd_index'],
        model_type=config['training_xlstm']['independent_multi_vd']['model_type'],
        experiment_name=os.path.basename(config['training_xlstm']['independent_multi_vd']['experiment_dir']),
        embedding_dim=config['training_xlstm']['independent_multi_vd']['embedding_dim'],
        num_blocks=config['training_xlstm']['independent_multi_vd']['num_blocks'],
        slstm_at=config['training_xlstm']['independent_multi_vd']['slstm_at']
    shell:
        """
        python scripts/train/without_social_pooling/train_independent_multi_vd.py \
        --data_path {input.h5_file} \
        --epochs {params.epochs} \
        --batch_size {params.batch_size} \
        --sequence_length {params.sequence_length} \
        --num_vds {params.num_vds} \
        --target_vd_index {params.target_vd_index} \
        --model_type {params.model_type} \
        --experiment_name {params.experiment_name} \
        --save_dir $(dirname {output.training_history}) \
        --embedding_dim {params.embedding_dim} \
        --num_blocks {params.num_blocks} \
        --slstm_at {params.slstm_at} >> {log} 2>&1
        """
```

### 性能比較規則

```python
# 新增 LSTM vs xLSTM 比較規則
rule compare_lstm_xlstm_single_vd:
    input:
        lstm_history=rules.train_single_vd_without_social_pooling.output.training_history,
        xlstm_history=rules.train_single_vd_xlstm.output.training_history
    output:
        comparison_report="blob/experiments/dev/comparison/single_vd_lstm_vs_xlstm.md",
        comparison_plots="blob/experiments/dev/comparison/single_vd_plots/"
    log:
        "logs/comparison/single_vd_lstm_vs_xlstm.log"
    shell:
        """
        python scripts/utils/compare_models.py \
        --lstm_experiment_dir $(dirname {input.lstm_history}) \
        --xlstm_experiment_dir $(dirname {input.xlstm_history}) \
        --output_report {output.comparison_report} \
        --output_plots_dir {output.comparison_plots} \
        --comparison_name "Single VD: LSTM vs xLSTM" >> {log} 2>&1
        """

rule compare_lstm_xlstm_multi_vd:
    input:
        lstm_history=rules.train_multi_vd_without_social_pooling.output.training_history,
        xlstm_history=rules.train_multi_vd_xlstm.output.training_history
    output:
        comparison_report="blob/experiments/dev/comparison/multi_vd_lstm_vs_xlstm.md",
        comparison_plots="blob/experiments/dev/comparison/multi_vd_plots/"
    log:
        "logs/comparison/multi_vd_lstm_vs_xlstm.log"
    shell:
        """
        python scripts/utils/compare_models.py \
        --lstm_experiment_dir $(dirname {input.lstm_history}) \
        --xlstm_experiment_dir $(dirname {input.xlstm_history}) \
        --output_report {output.comparison_report} \
        --output_plots_dir {output.comparison_plots} \
        --comparison_name "Multi-VD: LSTM vs xLSTM" >> {log} 2>&1
        """

rule compare_lstm_xlstm_independent_multi_vd:
    input:
        lstm_history=rules.train_independent_multi_vd_without_social_pooling.output.training_history,
        xlstm_history=rules.train_independent_multi_vd_xlstm.output.training_history
    output:
        comparison_report="blob/experiments/dev/comparison/independent_multi_vd_lstm_vs_xlstm.md",
        comparison_plots="blob/experiments/dev/comparison/independent_multi_vd_plots/"
    log:
        "logs/comparison/independent_multi_vd_lstm_vs_xlstm.log"  
    shell:
        """
        python scripts/utils/compare_models.py \
        --lstm_experiment_dir $(dirname {input.lstm_history}) \
        --xlstm_experiment_dir $(dirname {input.xlstm_history}) \
        --output_report {output.comparison_report} \
        --output_plots_dir {output.comparison_plots} \
        --comparison_name "Independent Multi-VD: LSTM vs xLSTM" >> {log} 2>&1
        """

# 綜合比較規則
rule compare_all_models:
    input:
        single_vd_comparison="blob/experiments/dev/comparison/single_vd_lstm_vs_xlstm.md",
        multi_vd_comparison="blob/experiments/dev/comparison/multi_vd_lstm_vs_xlstm.md", 
        independent_multi_vd_comparison="blob/experiments/dev/comparison/independent_multi_vd_lstm_vs_xlstm.md"
    output:
        comprehensive_report="blob/experiments/dev/comparison/comprehensive_model_comparison.md"
    log:
        "logs/comparison/comprehensive_comparison.log"
    shell:
        """
        python scripts/utils/generate_comprehensive_comparison.py \
        --input_reports {input.single_vd_comparison} {input.multi_vd_comparison} {input.independent_multi_vd_comparison} \
        --output_report {output.comprehensive_report} >> {log} 2>&1
        """
```

## 📊 使用流程設計

### 1. 單獨訓練 LSTM（現有功能）
```bash
# 訓練單一 LSTM 模型
snakemake train_single_vd_without_social_pooling

# 訓練所有 LSTM 模型
snakemake train_multi_vd_without_social_pooling train_independent_multi_vd_without_social_pooling
```

### 2. 單獨訓練 xLSTM（新功能）
```bash
# 訓練單一 xLSTM 模型
snakemake train_single_vd_xlstm

# 訓練所有 xLSTM 模型  
snakemake train_multi_vd_xlstm train_independent_multi_vd_xlstm
```

### 3. 並行訓練與比較（新功能）
```bash
# 同時訓練 LSTM 和 xLSTM，然後比較
snakemake compare_lstm_xlstm_single_vd

# 訓練所有模型並生成綜合比較報告
snakemake compare_all_models
```

### 4. 靈活的目標規則
```bash
# 只要 Single VD 的所有結果（LSTM + xLSTM + 比較）
snakemake compare_lstm_xlstm_single_vd

# 要所有模型的完整比較
snakemake compare_all_models

# 傳統方式：只要 LSTM 基準
snakemake generate_single_vd_report
```

## 🔄 配置文件更新策略

### 階段 1: 配置擴展
1. 在 `cfgs/snakemake/default.yaml` 中添加 `training_xlstm` 區塊
2. 保持現有 `training` 區塊完全不變  
3. 測試配置文件解析

### 階段 2: 規則添加
1. 在 Snakefile 中添加 xLSTM 訓練規則
2. 添加比較規則
3. 測試規則依賴關係

### 階段 3: 腳本修改
1. 修改訓練腳本支援 xLSTM 參數
2. 創建模型比較腳本
3. 測試端到端工作流程

## ⚠️ 注意事項

### 向後兼容性
- 現有的 `training` 配置保持完全不變
- 現有的 LSTM 訓練規則保持不變
- 可以繼續單獨使用 LSTM 工作流程

### 命名規範
- LSTM 實驗目錄: `blob/experiments/dev/single_vd_lstm`
- xLSTM 實驗目錄: `blob/experiments/dev/single_vd_xlstm`  
- 比較結果目錄: `blob/experiments/dev/comparison/`

### 參數調優
- xLSTM 可能需要較小的 batch_size（記憶體需求較高）
- embedding_dim 和 num_blocks 需要根據實際情況調整
- 初期使用 vanilla backend 確保穩定性

## 📋 實施檢查清單

- [ ] 創建 `training_xlstm` 配置區塊
- [ ] 修改 Snakefile 添加 xLSTM 訓練規則
- [ ] 創建模型比較腳本
- [ ] 修改訓練腳本支援 xLSTM 參數
- [ ] 測試配置文件解析
- [ ] 測試單一 xLSTM 訓練規則
- [ ] 測試比較規則
- [ ] 驗證向後兼容性
- [ ] 文檔化新的使用流程

---

**文檔狀態**: 設計階段  
**相關 ADR**: ADR-0501  
**下一步**: 實施配置文件修改
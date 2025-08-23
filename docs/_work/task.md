# Dynamic Model Configuration System Design

## Objective
Build a dynamic configuration system based on `model.name` that automatically determines required parameters and configuration structure based on model type, with proper separation of concerns between model architecture and training/data configurations.

## Design Concept

### Layered Configuration Architecture
Instead of mixing model architecture with training configurations, use a layered approach:

```
Model Architecture (cfgs/models/) + Data/Training Mode (cfgs/data/) = Complete Configuration
```

### Configuration File Structure

**Pure Model Architecture Configuration:**
```yaml
# cfgs/models/xlstm_base.yaml
model:
  name: "TrafficXLSTM"
  xlstm:
    input_size: 3
    embedding_dim: 128
    hidden_size: 128
    num_blocks: 6
    slstm_at: [1, 3]
    slstm_backend: "vanilla"
    mlstm_backend: "vanilla"
    context_length: 256
    dropout: 0.1
    sequence_length: 12
    prediction_length: 1
    batch_first: true
```

**Data/Training Mode Configuration:**
```yaml
# cfgs/data/multi_vd_5.yaml
data:
  multi_vd_mode: true
  num_vds: 5

# Override model input_size for multi-VD mode
model:
  xlstm:
    input_size: 15  # 3 * 5 VDs
```

### Supported Model Types
1. **TrafficLSTM** - Traditional LSTM model
2. **TrafficXLSTM** - Extended LSTM model  
3. **Transformer** - Transformer architecture
4. **Future Extensions** - Social Pooling, etc.

## Implementation Architecture

### 1. Model Registry System
```python
# src/social_xlstm/config/registry.py
MODEL_REGISTRY = {
    "TrafficLSTM": {
        "config_class": TrafficLSTMConfig,
        "model_class": TrafficLSTM,
        "config_key": "lstm"
    },
    "TrafficXLSTM": {
        "config_class": TrafficXLSTMConfig, 
        "model_class": TrafficXLSTM,
        "config_key": "xlstm"
    },
    "Transformer": {
        "config_class": TransformerConfig,
        "model_class": TransformerModel,
        "config_key": "transformer"
    }
}
```

### 2. Configuration Manager with Layered Support
```python
# src/social_xlstm/config/manager.py
class DynamicModelConfigManager:
    @classmethod
    def from_merged_config(cls, merged_config_dict: dict):
        """Load configuration from pre-merged configuration dict"""
        model_name = merged_config_dict['model']['name']
        
        # Get corresponding configuration class from registry
        registry_info = MODEL_REGISTRY[model_name]
        config_class = registry_info['config_class']
        config_key = registry_info['config_key']
        
        # Extract model-specific parameters
        model_params = merged_config_dict['model'][config_key]
        
        return config_class(**model_params)
    
    @classmethod
    def from_yaml_files(cls, yaml_paths: list):
        """Load and merge multiple YAML files using snakemake_warp logic"""
        # Use existing snakemake_warp.py merge_configs function
        merged_config = merge_configs(yaml_paths)
        return cls.from_merged_config(merged_config)
```

### 3. YAML Template Examples

**TrafficLSTM Base Configuration:**
```yaml
# cfgs/models/lstm_base.yaml
model:
  name: "TrafficLSTM"
  lstm:
    input_size: 3
    hidden_size: 128
    num_layers: 2
    dropout: 0.2
    sequence_length: 12
    prediction_length: 1
    batch_first: true
    bidirectional: false
```

**TrafficXLSTM Base Configuration:**
```yaml
# cfgs/models/xlstm_base.yaml
model:
  name: "TrafficXLSTM"
  xlstm:
    input_size: 3
    embedding_dim: 128
    hidden_size: 128
    num_blocks: 6
    slstm_at: [1, 3]
    slstm_backend: "vanilla"
    mlstm_backend: "vanilla"
    context_length: 256
    dropout: 0.1
    sequence_length: 12
    prediction_length: 1
    batch_first: true
```

**Single VD Mode:**
```yaml
# cfgs/data/single_vd.yaml
data:
  multi_vd_mode: false
  num_vds: 1
```

**Multi VD Mode:**
```yaml
# cfgs/data/multi_vd_5.yaml
data:
  multi_vd_mode: true
  num_vds: 5

# Override model configuration for multi-VD
model:
  lstm:
    input_size: 15  # 3 * 5 VDs for LSTM
  xlstm:
    input_size: 15  # 3 * 5 VDs for xLSTM
```

### 4. Training Script Integration

**Using snakemake_warp.py for configuration merging:**
```bash
# Merge model + data configurations
python workflow/snakemake_warp.py \
  --configfile cfgs/models/xlstm_base.yaml \
  --configfile cfgs/data/multi_vd_5.yaml \
  train_model
```

**In Python training script:**
```python
# Usage example
from social_xlstm.config import DynamicModelConfigManager
from workflow.snakemake_warp import merge_configs

def main():
    # Option 1: Use pre-merged config from snakemake_warp
    config = DynamicModelConfigManager.from_merged_config(snakemake.config)
    
    # Option 2: Merge configs directly in Python
    config_files = [args.model_config, args.data_config]
    config = DynamicModelConfigManager.from_yaml_files(config_files)
    
    model = config.create_model()
```

## Directory Structure

```
cfgs/
├── models/                 # Pure model architecture configs
│   ├── lstm_base.yaml
│   ├── xlstm_base.yaml
│   └── transformer_base.yaml
├── data/                   # Data and training mode configs
│   ├── single_vd.yaml
│   ├── multi_vd_3.yaml
│   └── multi_vd_5.yaml
├── training/              # Training hyperparameters (optional)
│   ├── default.yaml
│   └── fast_dev.yaml
└── experiments/           # Complete experiment configs (pre-combined for common scenarios)
    ├── lstm_single_baseline.yaml      # LSTM + single VD + default training
    ├── xlstm_multi_vd_experiment.yaml # xLSTM + multi VD + specific experiment settings
    └── transformer_comparison.yaml    # Transformer baseline for comparison
```

## Purpose of Experiments Directory

The `experiments/` directory serves several key purposes:

### 1. **Rapid Experiment Launch**
Pre-configured files for common experimental scenarios:
```bash
# Quick start - no need to manually combine multiple configs
python workflow/snakemake_warp.py --configfile cfgs/experiments/lstm_single_baseline.yaml train_model
```

### 2. **Reproducible Research**
Each experiment file represents a complete, self-contained configuration:
```yaml
# cfgs/experiments/lstm_single_baseline.yaml
includes:
  - "../models/lstm_base.yaml"      # Base model architecture
  - "../data/single_vd.yaml"       # Single VD mode
  - "../training/default.yaml"     # Standard training params

# Experiment-specific overrides
experiment:
  name: "LSTM_Single_VD_Baseline"
  description: "Baseline LSTM performance on single VD traffic prediction"
  tags: ["baseline", "lstm", "single-vd"]

training:
  epochs: 100
  batch_size: 64
  save_checkpoint_every: 10
```

### 3. **Paper/Publication Support** 
Easy reference for reproducing published results:
- `lstm_paper_baseline.yaml` - Exact configuration from Paper A
- `xlstm_comparison.yaml` - Setup for comparing xLSTM vs LSTM
- `ablation_study_*.yaml` - Various ablation study configurations

### 4. **Team Collaboration**
Standardized experiments that team members can easily run:
- New team members can quickly understand project capabilities
- Consistent experimental setups across different researchers
- Easy comparison of results

### 5. **CI/CD Integration**
Automated testing of key experimental configurations:
```yaml
# In CI pipeline
- name: Test baseline experiments
  run: |
    python workflow/snakemake_warp.py --configfile cfgs/experiments/quick_smoke_test.yaml train_model
```

## Advantages

1. **Clear Separation of Concerns** - Model architecture vs training/data mode
2. **High Reusability** - Same model config can be used with different VD modes
3. **Leverages Existing Tools** - Uses snakemake_warp.py's deep_merge functionality
4. **Type Safety** - Each model has dedicated configuration class
5. **Extensible** - Easy to add new models or data modes
6. **Parameter Validation** - Independent validation logic for each component

## Implementation Steps

1. **Refactor Configuration Classes** - Separate data/training concerns from model configs
2. **Build Model Registry System** - Define supported model types
3. **Create Layered Configuration Manager** - Support merged configuration loading
4. **Design YAML Template Structure** - Create standard templates for each layer
5. **Update Training Scripts** - Support layered configuration loading
6. **Integration with snakemake_warp.py** - Ensure smooth configuration merging
7. **Test and Validation** - Comprehensive testing of all combinations

## Configuration Merging Rules

Using snakemake_warp.py's deep_merge functionality:

1. **Priority Order**: Base model config < Data mode config < Training config < CLI overrides
2. **Merge Semantics**: 
   - Dictionaries: Deep merge (later overwrites earlier)
   - Lists/scalars: Complete replacement
3. **Validation**: Ensure data mode configs don't accidentally override critical model architecture parameters
4. **Safety**: Generate resolved_config.yaml for full traceability

## Benefits of This Approach

1. **Modularity** - Mix and match model architectures with different training modes
2. **Maintainability** - Clear responsibility boundaries between configuration types
3. **Flexibility** - Same model can easily switch between single-VD and multi-VD modes
4. **Consistency** - Standardized configuration structure across all model types
5. **Tool Reuse** - Leverages existing snakemake_warp.py infrastructure
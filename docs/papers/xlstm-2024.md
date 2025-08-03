# xLSTM: Extended Long Short-Term Memory

**Authors**: Maximilian Beck, Korbinian Pöppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael Kopp, Günter Klambauer, Johannes Brandstetter, Sepp Hochreiter  
**Affiliation**: NXAI GmbH, Johannes Kepler University Linz  
**Publication**: NeurIPS 2024 (Spotlight Paper)  
**arXiv**: https://arxiv.org/abs/2405.04517  
**GitHub**: https://github.com/NX-AI/xlstm  
**Submitted**: May 7, 2024  
**Last Revised**: December 6, 2024

---

## Abstract

In the 1990s, the constant error carousel and gating of the Long Short-Term Memory (LSTM) were introduced as the remedy to the vanishing gradient problem of Recurrent Neural Networks (RNNs). However, while these were effective for sequence modeling at their time, they have since lost the race to Transformers. Even though Transformers provide more performance for sequence modeling, their O(T²) attention mechanism is computationally intensive compared to LSTMs' O(T) complexity. In this work, we revisit the LSTM and propose extensions to address its limitations. Specifically, we introduce exponential gating and modify the memory structure to unlock the LSTM's potential. These modifications, termed "eXtended LSTM" (xLSTM), enable the scaling of LSTMs to billions of parameters and demonstrate competitive performance against modern architectures like Transformers and State Space Models.

## 1. Introduction

The Long Short-Term Memory (LSTM) network was a revolutionary advancement in Recurrent Neural Networks (RNNs), introduced to solve the vanishing gradient problem through constant error carousels and gating mechanisms. However, with the rise of Transformers, LSTMs have largely been relegated to niche applications despite their computational efficiency advantages.

This paper explores the question: **"How far can we scale LSTMs when equipped with modern training techniques and architectural modifications?"**

## 2. Key Innovations

### 2.1 Exponential Gating

Traditional LSTM gates use sigmoid activation functions, which can lead to saturation and gradient flow issues. xLSTM introduces **exponential gating** with appropriate normalization to:

- Improve gradient flow through time
- Enable more expressive gating mechanisms
- Maintain numerical stability through careful normalization

### 2.2 Modified Memory Structures

xLSTM introduces two complementary memory architectures:

#### Scalar LSTM (sLSTM)
- **Scalar Memory**: Traditional cell state structure with enhanced capabilities
- **Scalar Update**: Modified update rules for better information flow
- **Memory Mixing**: New mechanisms for combining past and present information

#### Matrix LSTM (mLSTM)
- **Matrix Memory**: Full matrix storage for richer representational capacity
- **Covariance Update Rule**: Advanced update mechanisms based on covariance structures
- **Full Parallelization**: Can be computed in parallel, similar to Transformers

## 3. Architecture Details

### 3.1 sLSTM (Scalar LSTM)

The sLSTM extends traditional LSTM with the following modifications:

#### Exponential Gates
```
i_t = exp(W_i x_t + U_i h_{t-1} + b_i)  # Input gate (exponential)
f_t = σ(W_f x_t + U_f h_{t-1} + b_f)    # Forget gate (sigmoid)
o_t = σ(W_o x_t + U_o h_{t-1} + b_o)    # Output gate (sigmoid)
```

#### Memory Cell with Normalization
```
c̃_t = tanh(W_c x_t + U_c h_{t-1} + b_c)
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
n_t = f_t ⊙ n_{t-1} + i_t               # Normalizer state
h_t = o_t ⊙ (c_t / n_t)                 # Normalized hidden state
```

Key Features:
- **Exponential input gating** for improved expressiveness
- **Normalizer state** to prevent overflow and maintain stability
- **Enhanced memory mixing** through modified update rules

### 3.2 mLSTM (Matrix LSTM)

The mLSTM introduces matrix-valued memory for increased capacity:

#### Matrix Memory Structure
```
C_t ∈ ℝ^{d×d}    # Matrix memory
n_t ∈ ℝ^d        # Normalizer vector
```

#### Covariance Update Rule
```
K_t = W_k x_t + U_k h_{t-1}  # Key vector
V_t = W_v x_t + U_v h_{t-1}  # Value vector
Q_t = W_q x_t                # Query vector

C_t = f_t ⊙ C_{t-1} + i_t ⊙ (V_t K_t^T)  # Matrix update
n_t = f_t ⊙ n_{t-1} + i_t ⊙ K_t          # Vector update
h_t = o_t ⊙ (C_t Q_t / max(n_t^T Q_t, 1)) # Output computation
```

Key Features:
- **Matrix memory** for richer representational capacity
- **Attention-like mechanics** through query-key-value interactions
- **Parallel computation** capabilities similar to Transformers
- **Covariance-based updates** for sophisticated memory mechanisms

### 3.3 xLSTM Block Architecture

xLSTM blocks integrate sLSTM and mLSTM into residual architectures:

```
Block Structure:
Input → LayerNorm → sLSTM/mLSTM → Residual Connection → Output
      ↘                                                 ↗
        ──────────── Skip Connection ─────────────────
```

#### Residual Integration
- **Pre-LayerNorm**: Applied before LSTM processing
- **Residual connections**: Enable deep stacking
- **Block stacking**: Multiple xLSTM blocks can be stacked
- **Mixed architectures**: Combining sLSTM and mLSTM blocks

## 4. Computational Complexity

### Complexity Comparison

| Model | Per-Step Complexity | Parallelization |
|-------|-------------------|-----------------|
| Traditional LSTM | O(d²) | Sequential |
| sLSTM | O(d²) | Sequential |
| mLSTM | O(d³) | Parallel |
| Transformer | O(T²d) | Parallel |

### Memory Efficiency
- **sLSTM**: Similar memory requirements to traditional LSTM
- **mLSTM**: Higher memory due to matrix storage (O(d²))
- **Training**: Both variants support gradient checkpointing
- **Inference**: Linear complexity in sequence length

## 5. Experimental Results

### Language Modeling Performance
- **7B Parameter Model**: Trained on 2.3T tokens
- **Competitive Performance**: Matches Transformer baselines
- **Scaling Properties**: Favorable scaling laws up to billions of parameters

### Task-Specific Benchmarks
- **Parity Tasks**: Superior performance on algorithmic reasoning
- **Multi-Query Associative Recall**: Enhanced memory capabilities
- **Long Context**: Better handling of extended sequences

### Efficiency Metrics
- **Training Speed**: Comparable to Transformers for moderate lengths
- **Inference Efficiency**: Significant advantages for long sequences
- **Memory Usage**: Lower memory footprint during inference

## 6. Advantages Over Traditional LSTM

### Enhanced Expressiveness
- **Exponential gating** provides more flexible control mechanisms
- **Matrix memory** enables richer representational capacity
- **Improved gradient flow** through better normalization

### Scalability
- **Billion-parameter models** successfully trained
- **Modern optimization techniques** fully compatible
- **Distributed training** support through proper implementation

### Parallelization
- **mLSTM parallelization** offers computational advantages
- **Mixed architectures** balance efficiency and expressiveness
- **Hardware optimization** through CUDA kernels

## 7. Implementation Details

### Software Requirements
- **PyTorch**: Version >=1.8
- **CUDA Support**: Custom kernels for acceleration
- **Hardware**: GPU recommended for large models

### Configuration Options
```python
# sLSTM Configuration
slstm_config = {
    "input_size": 512,
    "hidden_size": 1024,
    "num_layers": 12,
    "bias": True,
    "batch_first": True
}

# mLSTM Configuration  
mlstm_config = {
    "input_size": 512,
    "hidden_size": 1024,
    "num_layers": 6,
    "bias": True,
    "batch_first": True
}
```

### Training Considerations
- **Learning Rate Scheduling**: Cosine annealing recommended
- **Gradient Clipping**: Essential for stability
- **Mixed Precision**: FP16 training supported
- **Checkpointing**: For memory-efficient training

## 8. Impact on Social-xLSTM Project

### Direct Applications

#### Enhanced Temporal Modeling
- **Superior Memory**: Matrix memory enables better traffic pattern capture
- **Long-term Dependencies**: Improved modeling of temporal relationships
- **Scalability**: Support for larger VD networks

#### Architecture Integration
```python
# Social-xLSTM Integration
class DistributedSocialXLSTMModel:
    def __init__(self, xlstm_config):
        # Per-VD xLSTM instances with enhanced capabilities
        self.vd_manager = VDXLSTMManager(xlstm_config)
        # sLSTM for temporal patterns, mLSTM for complex interactions
        self.temporal_xlstm = sLSTM(xlstm_config)
        self.interaction_xlstm = mLSTM(xlstm_config)
```

#### Performance Benefits
- **Better Accuracy**: Enhanced memory improves prediction quality
- **Training Efficiency**: Parallel mLSTM reduces training time
- **Scalability**: Support for larger traffic networks

### Technical Adaptations

#### VD-Specific Optimizations
- **sLSTM for Temporal Sequences**: Individual VD traffic patterns
- **mLSTM for Social Interactions**: Complex multi-VD relationships
- **Hybrid Architecture**: Combining both variants optimally

#### Memory Management
- **Dynamic Allocation**: Efficient VD instance management
- **Gradient Checkpointing**: Memory-efficient training
- **Mixed Precision**: Accelerated computation

## 9. Future Research Directions

### Architectural Extensions
- **Attention-augmented xLSTM**: Combining with attention mechanisms
- **Hierarchical xLSTM**: Multi-scale temporal modeling
- **Sparse xLSTM**: Efficient processing for large networks

### Application Domains
- **Time Series Forecasting**: Beyond language modeling
- **Multimodal Processing**: Integrating multiple data types
- **Real-time Systems**: Low-latency inference applications

## References

```bibtex
@article{beck2024xlstm,
  title={xLSTM: Extended Long Short-Term Memory},
  author={Beck, Maximilian and Pöppel, Korbinian and Spanring, Markus and Auer, Andreas and Prudnikova, Oleksandra and Kopp, Michael and Klambauer, Günter and Brandstetter, Johannes and Hochreiter, Sepp},
  journal={arXiv preprint arXiv:2405.04517},
  year={2024}
}
```

### Related Work
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation.
- Vaswani, A., et al. (2017). Attention is all you need. NeurIPS.
- Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces.

---

**Note**: This document summarizes the key innovations and technical details of xLSTM that are most relevant to the Social-xLSTM project. For complete mathematical formulations and experimental details, please refer to the original paper and official repository.
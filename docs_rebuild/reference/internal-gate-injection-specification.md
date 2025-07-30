# Internal Gate Injection (IGI) Specification

Complete technical specification for Social-xLSTM with Internal Gate Injection strategy.

## Overview

Internal Gate Injection (IGI) is an advanced integration strategy that directly injects social pooling information into the gate computations of xLSTM blocks. Unlike Post-Fusion approaches that combine features after model processing, IGI enables deep spatial-temporal coupling by influencing memory operations at the gate level.

### Key Advantages

- **Deep Integration**: Social information participates in every memory update step
- **Fine-grained Control**: Precise modulation of input, forget, and output gates
- **Immediate Response**: Neighbor changes instantly reflected in memory states
- **Spatial-Temporal Coupling**: True fusion of spatial and temporal modeling

## Mathematical Formulation

### Notation

For a system with $N$ agents at time step $t$:

- $x_t^i \in \mathbb{R}^{d_{\text{in}}}$: Input feature vector for agent $i$
- $h_{t-1}^i \in \mathbb{R}^{d_h}$: Hidden state of agent $i$ at time $t-1$ (sLSTM only)
- $\mathbf{c}_i = (x_i, y_i)$: Spatial coordinates of agent $i$
- $\mathcal{N}_i = \{j : \|\mathbf{c}_j - \mathbf{c}_i\| \leq R, j \neq i\}$: Neighbor set of agent $i$
- $S_t^i \in \mathbb{R}^{d_s}$: Social pooling vector for agent $i$

### Social Pooling Vector Computation

The social pooling vector $S_t^i$ aggregates spatial information from neighboring agents:

$$S_t^i = \text{SocialPool}(\{h_{t-1}^j : j \in \mathcal{N}_i \cup \{i\}\}, \{\mathbf{c}_j : j \in \mathcal{N}_i \cup \{i\}\})$$

Where the pooling operation uses coordinate-driven spatial weights:

$$w_{ij} = \exp\left(-\frac{\|\mathbf{c}_i - \mathbf{c}_j\|^2}{2\sigma^2}\right)$$

$$S_t^i = \phi_{\text{soc}}\left(\sum_{j \in \mathcal{N}_i \cup \{i\}} \frac{w_{ij}}{\sum_{k} w_{ik}} h_{t-1}^j\right)$$

Where $\phi_{\text{soc}}$ is a learned transformation network:

$$S_t = \phi_{\text{soc}}(\mathbf{h}_{\text{flat}}) = \text{ReLU}(\mathbf{W}_2 \text{ReLU}(\mathbf{W}_1 \mathbf{h}_{\text{flat}} + \mathbf{b}_1) + \mathbf{b}_2)$$

## sLSTM with Internal Gate Injection

### Enhanced Gate Computation

Standard sLSTM gate computations are extended to include Social Pooling information:

$$\begin{align}
\tilde{i}_t &= \mathbf{W}_i^{(x)} x_t + \mathbf{R}_i h_{t-1} + \mathbf{U}_i S_t + b_i \\
\tilde{f}_t &= \mathbf{W}_f^{(x)} x_t + \mathbf{R}_f h_{t-1} + \mathbf{U}_f S_t + b_f \\
\tilde{z}_t &= \mathbf{W}_z^{(x)} x_t + \mathbf{R}_z h_{t-1} + \mathbf{U}_z S_t + b_z \\
\tilde{o}_t &= \mathbf{W}_o^{(x)} x_t + \mathbf{R}_o h_{t-1} + \mathbf{U}_o S_t + b_o
\end{align}$$

Where:
- $\mathbf{W}_{\bullet}^{(x)} \in \mathbb{R}^{d_h \times d_{\text{in}}}$: Input weight matrices
- $\mathbf{R}_{\bullet} \in \mathbb{R}^{d_h \times d_h}$: Recurrent weight matrices  
- $\mathbf{U}_{\bullet} \in \mathbb{R}^{d_h \times d_s}$: **Social Pooling weight matrices (new)**
- $b_{\bullet} \in \mathbb{R}^{d_h}$: Bias vectors

### Exponential Gating and Memory Update

Applying xLSTM's exponential gating mechanism:

$$\begin{align}
i_t &= \exp(\tilde{i}_t) \\
f_t &= \sigma(\tilde{f}_t) \\
z_t &= \tanh(\tilde{z}_t) \\
o_t &= \sigma(\tilde{o}_t)
\end{align}$$

Memory state update and output computation:

$$\begin{align}
c_t &= f_t \odot c_{t-1} + i_t \odot z_t \\
n_t &= f_t \odot n_{t-1} + i_t \\
\tilde{h}_t &= \frac{c_t}{n_t} \\
h_t &= o_t \odot \tilde{h}_t
\end{align}$$

### Social Influence Analysis

Social Pooling vector $S_t$ influences each gate mechanism:

- **Input Gate $i_t$**: Controls intensity of new information (including social information) writing
- **Forget Gate $f_t$**: Determines whether to retain or forget social context-related memories
- **Candidate Values $z_t$**: Directly influences memory content, enabling social modulation
- **Output Gate $o_t$**: Controls the degree of social-aware memory output

This design enables fine-grained social influence over memory *writing*, *forgetting*, and *output* processes.

## mLSTM with Internal Gate Injection

### Query, Key, Value Computation with Social Integration

mLSTM's query, key, value computations integrate Social Pooling information:

$$\begin{align}
q_t &= \mathbf{W}_q^{(x)} x_t + \mathbf{U}_q S_t + b_q \\
k_t &= \frac{1}{\sqrt{d_k}} \left(\mathbf{W}_k^{(x)} x_t + \mathbf{U}_k S_t\right) + b_k \\
v_t &= \mathbf{W}_v^{(x)} x_t + \mathbf{U}_v S_t + b_v
\end{align}$$

Where $d_k$ is the key vector dimension, and the scaling factor $1/\sqrt{d_k}$ stabilizes training.

### Social-Aware Gate Mechanisms

mLSTM gate computations also integrate social information:

$$\begin{align}
i_t &= \exp(\mathbf{w}_i^{\top} x_t + \mathbf{u}_i^{\top} S_t + b_i) \\
f_t &= \sigma(\mathbf{w}_f^{\top} x_t + \mathbf{u}_f^{\top} S_t + b_f) \\
o_t &= \sigma(\mathbf{w}_o^{\top} x_t + \mathbf{u}_o^{\top} S_t + b_o)
\end{align}$$

Note that in mLSTM, gate weights $\mathbf{w}_{\bullet}, \mathbf{u}_{\bullet}$ are vectors rather than matrices, reflecting mLSTM's architectural characteristics.

### Matrix Memory Update and Retrieval

Matrix memory update and retrieval processes:

$$\begin{align}
\mathbf{C}_t &= f_t \mathbf{C}_{t-1} + i_t v_t k_t^{\top} \\
\mathbf{n}_t &= f_t \mathbf{n}_{t-1} + i_t k_t \\
h_t &= o_t \odot \frac{\mathbf{C}_t q_t}{\max\{|\mathbf{n}_t^{\top} q_t|, 1\}}
\end{align}$$

Where:
- $\mathbf{C}_t \in \mathbb{R}^{d_h \times d_h}$: Matrix memory state
- $\mathbf{n}_t \in \mathbb{R}^{d_h}$: Normalization vector
- The $\max$ operation in the denominator prevents numerical instability

## Computational Complexity Analysis

### Social Pooling Complexity

For each agent $i$, Social Pooling computational complexity includes:

#### Neighbor Discovery and Grid Construction
$$\begin{align}
\mathcal{O}_{\text{neighbor}} &= \mathcal{O}(|\mathcal{N}_i|) \approx \mathcal{O}(\pi R^2 \rho) \\
\mathcal{O}_{\text{grid}} &= \mathcal{O}(|\mathcal{N}_i| \cdot d_h) \approx \mathcal{O}(\pi R^2 \rho \cdot d_h)
\end{align}$$

Where $\rho$ is agent density.

#### Feature Embedding
$$\mathcal{O}_{\text{embedding}} = \mathcal{O}(M \cdot N \cdot d_h \cdot d_{\text{mid}} + d_{\text{mid}} \cdot d_s)$$

### sLSTM with IGI Complexity

Per-timestep sLSTM computational complexity:

#### Standard sLSTM Operations
$$\mathcal{O}_{\text{sLSTM-std}} = \mathcal{O}(d_{\text{in}} \cdot d_h + d_h^2)$$

#### Social Integration Overhead
$$\mathcal{O}_{\text{sLSTM-social}} = \mathcal{O}(4 \cdot d_s \cdot d_h)$$

#### Total Complexity
$$\mathcal{O}_{\text{sLSTM-IGI}} = \mathcal{O}(d_{\text{in}} \cdot d_h + d_h^2 + d_s \cdot d_h)$$

### mLSTM with IGI Complexity

#### Standard mLSTM Operations
$$\mathcal{O}_{\text{mLSTM-std}} = \mathcal{O}(d_{\text{in}} \cdot d_h + d_h^3)$$

#### Social Integration Overhead
$$\mathcal{O}_{\text{mLSTM-social}} = \mathcal{O}(3 \cdot d_s \cdot d_h + 3 \cdot d_s)$$

#### Total Complexity
$$\mathcal{O}_{\text{mLSTM-IGI}} = \mathcal{O}(d_{\text{in}} \cdot d_h + d_h^3 + d_s \cdot d_h)$$

### Complexity Comparison

| Operation | Post-Fusion | Internal Injection |
|-----------|-------------|-------------------|
| Base Model | $\mathcal{O}(d_h^3)$ | $\mathcal{O}(d_h^3)$ |
| Social Pooling | $\mathcal{O}(N \cdot \bar{k} \cdot d_h)$ | $\mathcal{O}(N \cdot \bar{k} \cdot d_h)$ |
| Integration | $\mathcal{O}(d_h \cdot d_s)$ | $\mathcal{O}(d_h \cdot d_s)$ |
| **Total** | $\mathcal{O}(d_h^3 + N \cdot \bar{k} \cdot d_h)$ | $\mathcal{O}(d_h^3 + N \cdot \bar{k} \cdot d_h)$ |

Where $\bar{k}$ is the average number of neighbors per node.

## Hybrid xLSTM Block Architecture

Social-xLSTM with IGI uses mixed block structure combining sLSTM and mLSTM advantages:

### Block Structure

```
Input: x_t, S_t
   ↓
[LayerNorm] → [sLSTM with IGI] → [Residual]
   ↓                                ↓
[LayerNorm] → [mLSTM with IGI] → [Add] → Output
```

### Algorithm: Social-xLSTM IGI Block

```
Input: x_t, S_t, (hidden states from previous timestep)
Output: h_t

1. // Phase 1: Layer Normalization
   x_norm = LayerNorm(x_t)

2. // Phase 2: sLSTM Processing with IGI
   h_slstm = sLSTM_IGI(x_norm, S_t, h_slstm_prev)
   h_residual = x_norm + h_slstm

3. // Phase 3: mLSTM Processing with IGI  
   h_norm = LayerNorm(h_residual)
   h_mlstm = mLSTM_IGI(h_norm, S_t, C_mlstm_prev, n_mlstm_prev)
   h_t = h_residual + h_mlstm

4. Return h_t
```

### Complete Model Architecture

Complete Social-xLSTM IGI model contains $L$ stacked blocks:

$$\begin{align}
\mathbf{h}^{(0)} &= \text{InputEmbedding}(x_t) \\
\mathbf{h}^{(\ell)} &= \text{SocialBlock}_\ell(\mathbf{h}^{(\ell-1)}, S_t), \quad \ell = 1, \ldots, L \\
\mathbf{y}_t &= \text{OutputLayer}(\mathbf{h}^{(L)})
\end{align}$$

## Implementation Considerations

### Memory-Efficient Implementation

For large-scale deployments, consider:

1. **Sparse Social Pooling**: Limit neighbors using top-k selection
2. **Gradient Checkpointing**: Trade computation for memory by recomputing activations
3. **Mixed Precision**: Use FP16 for forward pass, FP32 for parameter updates

### Numerical Stability

#### Exponential Gates (sLSTM)
Use numerical stability measures:

$$\exp(x) = \begin{cases}
\exp(x) & \text{if } x \leq 10 \\
\exp(10) & \text{if } x > 10
\end{cases}$$

#### Matrix Memory (mLSTM)  
Normalize operations to prevent overflow:

$$\mathbf{C}_t = \frac{\mathbf{C}_t}{\|\mathbf{C}_t\|_F + \epsilon}$$

Where $\|\cdot\|_F$ is the Frobenius norm and $\epsilon = 1e-8$.

## Comparison with Post-Fusion Strategy

### Response Characteristics

**Memory Evolution Comparison**:

Let $\mathbf{M}_{\text{PF}}(t)$ and $\mathbf{M}_{\text{IGI}}(t)$ represent memory states at time $t$:

**Post-Fusion**:
$$\mathbf{M}_{\text{PF}}(t) = f_{\text{xLSTM}}(\mathbf{M}_{\text{PF}}(t-1), x_t)$$
$$\hat{y}_t = g_{\text{fusion}}(f_{\text{output}}(\mathbf{M}_{\text{PF}}(t)), S_t)$$

**Internal Gate Injection**:
$$\mathbf{M}_{\text{IGI}}(t) = f_{\text{xLSTM-IGI}}(\mathbf{M}_{\text{IGI}}(t-1), x_t, S_t)$$
$$\hat{y}_t = g_{\text{output}}(\mathbf{M}_{\text{IGI}}(t))$$

In IGI, social information $S_t$ directly participates in memory evolution, achieving deeper spatial-temporal coupling.

### Response Timing

- **Post-Fusion**: Response delay of 1 timestep  
- **IGI**: Immediate response, with response intensity related to gate values

This makes IGI particularly suitable for scenarios requiring rapid spatial adaptation.

## Advantages and Limitations

### IGI Method Advantages

1. **Deeper Integration**: Social information influences internal memory mechanisms
2. **Immediate Response**: Neighbor changes instantly affect memory states
3. **Fine-grained Control**: Precise modulation of each gate mechanism
4. **Enhanced Expressiveness**: Theoretical advantage in spatial-temporal modeling

### IGI Method Limitations

1. **Implementation Complexity**: More complex than Post-Fusion approaches
2. **Training Difficulty**: Requires careful hyperparameter tuning
3. **Computational Overhead**: Additional parameters and computations
4. **Debugging Challenges**: Internal integration makes issue diagnosis harder

## Selection Guidelines

**Choose IGI when**:
- High performance requirements
- Need for immediate spatial response  
- Research exploration of deep integration
- Sufficient computational resources

**Choose Post-Fusion when**:
- Rapid prototyping needs
- Stability is priority
- Limited computational resources
- Team collaborative development

## Related References

- **Post-Fusion Specification**: [post-fusion-specification.md](post-fusion-specification.md)
- **Mathematical Foundations**: [mathematical-specifications.md](mathematical-specifications.md)  
- **Implementation Guide**: [../how-to/use-social-pooling.md](../how-to/use-social-pooling.md)
- **Design Rationale**: [../explanation/social-pooling-design.md](../explanation/social-pooling-design.md)

---

*This specification provides the complete technical foundation for implementing Social-xLSTM with Internal Gate Injection strategy, including mathematical formulations, complexity analysis, and practical implementation guidance.*
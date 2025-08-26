# Social-xLSTM Mathematical Specifications

**Complete mathematical formulation and specifications for Social-xLSTM models.**

This document provides comprehensive mathematical definitions for the **Social-xLSTM** system, which combines:
- **xLSTM (Extended Long Short-Term Memory)** - Core innovation based on Beck et al. (2024)
- **Social Pooling** - Spatial aggregation mechanism based on Alahi et al. (2016) 
- **Distributed Architecture** - Each VD maintains independent xLSTM with shared weights

**Key Innovation**: The Social-xLSTM integrates xLSTM's sLSTM + mLSTM mixed architecture with coordinate-driven social pooling for traffic flow prediction.

## Problem Formulation

### Input Definition

For each vehicle detector (VD) node $i$ at time step $t$, the input feature vector is:

$$\mathbf{x}_i^t = [q_i^t, p_i^t, s_i^t, \ell_i, \mathbf{t}_i]^T$$

where:
- $q_i^t$ = current traffic volume at node $i$ at time $t$
- $p_i^t$ = current lane occupancy at node $i$ at time $t$  
- $s_i^t$ = current lane speed at node $i$ at time $t$
- $\ell_i$ = number of lanes at node $i$ (static)
- $\mathbf{t}_i$ = temporal features (hour, day, etc.)

Each node $i$ has fixed spatial coordinates $(x_i, y_i)$ representing its geographical location.

## Social-xLSTM Integration Architecture

**Social Pooling** operates on xLSTM hidden states rather than raw features, following the correct distributed architecture. The mathematical formulations below define the **Social-xLSTM** system with two integration strategies.

### Coordinate-Driven Spatial Aggregation

For target node $i$, we define the spatial neighborhood as:

$$\mathcal{N}_i = \{j : d(i,j) \leq R, j \neq i\}$$

where $d(i,j)$ is the distance between nodes $i$ and $j$, and $R$ is the pooling radius.

### Social-xLSTM Integration Strategies

**Post-Fusion Strategy** (Primary approach): Social pooling aggregates xLSTM hidden states:
$$\hat{y}_t^i = f_{\text{fusion}}(f_{\text{xLSTM}}(x_t^i), \text{SocialPool}(\{h_t^j\}, \{\mathbf{c}_j\}))$$

**Internal Gate Injection (IGI) Strategy**: Social information influences xLSTM gate computations:
$$h_t^i = f_{\text{xLSTM-IGI}}(x_t^i, \text{SocialPool}(\{h_{t-1}^j\}, \{\mathbf{c}_j\}), h_{t-1}^i)$$

Where $f_{\text{xLSTM}}$ represents the xLSTM processing (sLSTM + mLSTM blocks) and $h_t^j$ are xLSTM hidden states from neighboring VDs.

*For complete technical specifications of spatial pooling implementation, see:*
- [Social Pooling API Reference](../reference/api-reference.md#social-pooling-module)

### Distance Metrics

#### Euclidean Distance
$$d_{\text{euclidean}}(i,j) = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$$

#### Manhattan Distance  
$$d_{\text{manhattan}}(i,j) = |x_i - x_j| + |y_i - y_j|$$

#### Haversine Distance (for geographic coordinates)
$$d_{\text{haversine}}(i,j) = 2R_{\text{earth}} \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\phi_i)\cos(\phi_j)\sin^2\left(\frac{\Delta\lambda}{2}\right)}\right)$$

where $\phi$ represents latitude, $\lambda$ represents longitude, and $R_{\text{earth}}$ is Earth's radius.

### Spatial Weight Functions

#### Gaussian Kernel
$$w_{ij}^{\text{gaussian}} = \exp\left(-\frac{d(i,j)^2}{2\sigma^2}\right)$$

where $\sigma = R/3$ (pooling radius divided by 3).

#### Exponential Decay
$$w_{ij}^{\text{exponential}} = \exp\left(-\frac{d(i,j)}{\lambda}\right)$$

where $\lambda = R$ (pooling radius).

#### Linear Decay  
$$w_{ij}^{\text{linear}} = \max\left(0, 1 - \frac{d(i,j)}{R}\right)$$

#### Inverse Distance
$$w_{ij}^{\text{inverse}} = \frac{1}{1 + d(i,j)}$$

### Social-xLSTM Spatial Feature Aggregation

For node $i$ at time $t$, the spatially pooled xLSTM features are:

$$\mathbf{h}_i^{\text{social}} = \frac{\sum_{j \in \mathcal{N}_i \cup \{i\}} w_{ij} \mathbf{h}_j^{\text{xLSTM},t}}{\sum_{j \in \mathcal{N}_i \cup \{i\}} w_{ij}}$$

where $\mathbf{h}_j^{\text{xLSTM},t}$ is the xLSTM hidden state of node $j$ at time $t$, computed through the sLSTM + mLSTM mixed architecture.

**Key Distinction**: Social pooling operates on **xLSTM hidden states** (high-level learned representations) rather than raw input features, following the distributed Social-xLSTM architecture.

## Comparison with Original Social LSTM Implementation

### Implementation Divergence Analysis

This Social-xLSTM implementation **deviates significantly** from the original Social LSTM paper (Alahi et al., 2016) in the spatial aggregation methodology. This section provides mathematical comparison and justification.

### Original Social LSTM: Grid-Based Spatial Pooling

#### Grid Tensor Construction
The original Social LSTM constructs a spatial grid tensor for each agent $i$ at time $t$:

$$\mathbf{H}^i_t \in \mathbb{R}^{N_o \times N_o \times D}$$

where $N_o$ is the grid neighborhood size and $D$ is the hidden state dimension.

#### Grid-Based Aggregation Formula
$$\mathbf{H}^i_t(m, n, :) = \sum_{j \in \mathcal{N}_i} \mathbf{1}_{mn}[x^j_t - x^i_t, y^j_t - y^i_t] \cdot \mathbf{h}^j_{t-1}$$

where:
- $\mathbf{1}_{mn}[\cdot, \cdot]$ is an indicator function for grid cell $(m,n)$
- $\mathcal{N}_i$ is the set of neighboring agents
- $\mathbf{h}^j_{t-1}$ is the hidden state of agent $j$ at time $t-1$

#### Grid Cell Assignment
Relative positions are discretized into grid coordinates:
$$m = \lfloor \frac{x^j_t - x^i_t}{\Delta x} \rfloor + \frac{N_o}{2}$$
$$n = \lfloor \frac{y^j_t - y^i_t}{\Delta y} \rfloor + \frac{N_o}{2}$$

where $\Delta x$ and $\Delta y$ are the grid cell dimensions.

### Our Social-xLSTM: Distance-Based Continuous Pooling

#### Continuous Distance Computation
Instead of grid discretization, we use direct Euclidean distance:

$$d_{ij} = \|\mathbf{p}_i^t - \mathbf{p}_j^t\|_2 = \sqrt{(x_i^t - x_j^t)^2 + (y_i^t - y_j^t)^2}$$

#### Radius-Based Neighbor Selection
$$\mathcal{N}_i^R = \{j : d_{ij} \leq R, j \neq i\}$$

where $R$ is the interaction radius parameter.

#### Distance-Weighted Aggregation
$$\mathbf{h}_i^{\text{social}} = \frac{\sum_{j \in \mathcal{N}_i^R} w_{ij} \cdot \mathbf{h}_j^{\text{xLSTM},t}}{\sum_{j \in \mathcal{N}_i^R} w_{ij}}$$

where the weight function can be:
- **Uniform**: $w_{ij} = 1$
- **Inverse Distance**: $w_{ij} = \frac{1}{d_{ij} + \epsilon}$  
- **Gaussian**: $w_{ij} = \exp\left(-\frac{d_{ij}^2}{2\sigma^2}\right)$

### Mathematical Comparison Table

| Aspect | Original Social LSTM | Our Social-xLSTM |
|--------|---------------------|------------------|
| **Spatial Discretization** | $\mathbb{R}^2 \rightarrow \mathbb{Z}^2$ (grid cells) | $\mathbb{R}^2$ (continuous) |
| **Neighbor Definition** | $\mathbf{1}_{mn}$ grid indicator | $d_{ij} \leq R$ distance threshold |
| **Aggregation Tensor** | $\mathbf{H}^i_t \in \mathbb{R}^{N_o \times N_o \times D}$ | $\mathbf{h}_i^{\text{social}} \in \mathbb{R}^D$ |
| **Complexity** | $\mathcal{O}(N_o^2 \cdot D)$ | $\mathcal{O}(\|\mathcal{N}_i^R\| \cdot D)$ |
| **Hyperparameters** | Grid size $N_o$, cell size $\Delta x, \Delta y$ | Radius $R$, weight function |
| **Boundary Handling** | Hard grid boundaries | Smooth distance decay |

### Performance Implications

#### Computational Efficiency
- **Grid-based**: Requires $N_o^2$ grid cells even when sparse
- **Distance-based**: Scales with actual neighbor count $\|\mathcal{N}_i^R\|$

#### Representational Quality  
- **Grid-based**: Quantization artifacts at cell boundaries
- **Distance-based**: Smooth spatial gradients

#### Parameter Sensitivity
- **Grid-based**: Sensitive to grid resolution choice
- **Distance-based**: Intuitive radius parameter with physical meaning

### Justification for Distance-Based Approach

1. **Traffic Domain Suitability**: Vehicle detectors (VDs) are positioned at irregular geographic locations, not aligned to any natural grid structure.

2. **Computational Advantages**: For typical traffic densities, the number of neighbors within radius $R$ is much smaller than grid size $N_o^2$.

3. **Modern Alignment**: Contemporary trajectory prediction models (Social-GAN, Trajectron++) predominantly use distance-based continuous approaches.

4. **Gradient Quality**: Continuous distance functions provide smoother gradients for more stable training compared to discrete grid assignments.

### Implementation Notes

- **Backward Compatibility**: While mathematically different, both approaches achieve the same goal of spatial hidden state aggregation
- **Hyperparameter Translation**: Grid size $N_o$ roughly corresponds to $R/\text{typical\_VD\_spacing}$
- **Performance Baseline**: Direct comparison requires implementing both methods on identical datasets

**Reference**: See [ADR-001](../decisions/adr-001-distance-based-social-pooling.md) for detailed architectural decision rationale.

## Extended LSTM (xLSTM) Formulation - Core Innovation

**xLSTM** is the core innovation of the Social-xLSTM system, providing enhanced memory capabilities through two complementary architectures.

### Scalar-memory LSTM (sLSTM)

The sLSTM extends traditional LSTM with exponential gating and normalization for improved gradient flow and memory capacity.

#### Gate Computations
$$\begin{align}
\tilde{i}_t &= \mathbf{W}_i \mathbf{e}_t + \mathbf{R}_i \mathbf{h}_{t-1} + \mathbf{b}_i \\
\tilde{f}_t &= \mathbf{W}_f \mathbf{e}_t + \mathbf{R}_f \mathbf{h}_{t-1} + \mathbf{b}_f \\
\tilde{z}_t &= \mathbf{W}_z \mathbf{e}_t + \mathbf{R}_z \mathbf{h}_{t-1} + \mathbf{b}_z \\
\tilde{o}_t &= \mathbf{W}_o \mathbf{e}_t + \mathbf{R}_o \mathbf{h}_{t-1} + \mathbf{b}_o
\end{align}$$

#### Exponential Gating
$$\begin{align}
i_t &= \exp(\tilde{i}_t) \\
f_t &= \sigma(\tilde{f}_t) \\
z_t &= \tanh(\tilde{z}_t) \\
o_t &= \sigma(\tilde{o}_t)
\end{align}$$

#### Cell State and Hidden State Update
$$\begin{align}
c_t &= f_t \odot c_{t-1} + i_t \odot z_t \\
n_t &= f_t \odot n_{t-1} + i_t \\
\tilde{h}_t &= \frac{c_t}{n_t} \\
h_t &= o_t \odot \tilde{h}_t
\end{align}$$

where $n_t$ is the normalizer state to prevent overflow.

### Matrix-memory LSTM (mLSTM)

The mLSTM uses matrix-valued memory for higher capacity storage.

#### Query, Key, Value Computation
$$\begin{align}
\mathbf{q}_t &= \mathbf{W}_q \mathbf{e}_t \\
\mathbf{k}_t &= \mathbf{W}_k \mathbf{e}_t \\
\mathbf{v}_t &= \mathbf{W}_v \mathbf{e}_t
\end{align}$$

#### Matrix Memory Update
$$\begin{align}
\mathbf{C}_t &= \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot (\mathbf{v}_t \mathbf{k}_t^T) \\
\mathbf{n}_t &= \mathbf{f}_t \odot \mathbf{n}_{t-1} + \mathbf{i}_t \odot \mathbf{k}_t
\end{align}$$

where $\mathbf{C}_t \in \mathbb{R}^{d \times d}$ is the matrix memory and $\mathbf{n}_t \in \mathbb{R}^d$ is the normalization vector.

#### Memory Retrieval
$$\mathbf{h}_t = \mathbf{o}_t \odot \frac{\mathbf{C}_t \mathbf{q}_t}{\max(\mathbf{n}_t^T \mathbf{q}_t, 1)}$$

## Hybrid Social-xLSTM Block Stack Architecture

### Social-xLSTM Block Architecture

A **Social-xLSTM block** combines both sLSTM and mLSTM components with spatial awareness:

$$\text{Social-xLSTM-Block}(\mathbf{x}) = \text{LayerNorm}(\mathbf{x} + \text{mLSTM}(\text{sLSTM}(\mathbf{x})))$$

**Architecture Advantage**: The hybrid sLSTM + mLSTM design provides both efficient scalar memory (sLSTM) for temporal patterns and high-capacity matrix memory (mLSTM) for complex spatial-temporal interactions.

### Multi-Block Social-xLSTM Stack

The complete **Social-xLSTM model** consists of $L$ stacked blocks processing each VD independently:

$$\begin{align}
\mathbf{h}^{(0)}_i &= \text{InputEmbedding}(\mathbf{e}_{t,i}) \\
\mathbf{h}^{(\ell)}_i &= \text{Social-xLSTM-Block}_\ell(\mathbf{h}^{(\ell-1)}_i), \quad \ell = 1, \ldots, L \\
\mathbf{h}_{\text{social},i} &= \text{SocialPool}(\{\mathbf{h}^{(L)}_j\}_{j \in \mathcal{N}_i}) \\
\mathbf{y}_{t,i} &= \text{OutputLayer}([\mathbf{h}^{(L)}_i; \mathbf{h}_{\text{social},i}])
\end{align}$$

where $i$ indexes individual VDs and the final prediction combines both individual xLSTM processing and social spatial information.

## Social-xLSTM Loss Functions

### Traffic Prediction Loss for Social-xLSTM

For **Social-xLSTM traffic flow prediction**, we use a combination of regression losses optimized for xLSTM's enhanced representational capacity:

$$\mathcal{L}_{\text{Social-xLSTM}} = \alpha \mathcal{L}_{\text{MAE}} + \beta \mathcal{L}_{\text{MSE}} + \gamma \mathcal{L}_{\text{MAPE}}$$

where:

#### Mean Absolute Error (MAE)
$$\mathcal{L}_{\text{MAE}} = \frac{1}{N} \sum_{i=1}^N |\hat{y}_i - y_i|$$

#### Mean Squared Error (MSE)  
$$\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2$$

#### Mean Absolute Percentage Error (MAPE)
$$\mathcal{L}_{\text{MAPE}} = \frac{1}{N} \sum_{i=1}^N \left|\frac{\hat{y}_i - y_i}{y_i}\right|$$

### Social-xLSTM Regularization Terms

#### L2 Weight Decay for xLSTM Parameters
$$\mathcal{L}_{\text{L2-xLSTM}} = \lambda \sum_{\theta \in \Theta_{\text{xLSTM}}} \theta^2$$

where $\Theta_{\text{xLSTM}}$ includes both **sLSTM scalar parameters** and **mLSTM matrix parameters**.

#### Dropout Applied to xLSTM Hidden States
$$\mathbf{h}_{\text{xLSTM-dropout}} = \mathbf{h}_{\text{xLSTM}} \odot \mathbf{m}$$

where $\mathbf{m}$ is a binary mask with probability $p$ of being 0, applied to **xLSTM output states** before Social Pooling.

## Social-xLSTM Evaluation Metrics

### Regression Metrics for Social-xLSTM

#### Root Mean Squared Error (RMSE) for Social-xLSTM Predictions
$$\text{RMSE}_{\text{Social-xLSTM}} = \sqrt{\frac{1}{N} \sum_{i=1}^N (\hat{y}_{\text{xLSTM},i} - y_i)^2}$$

where $\hat{y}_{\text{xLSTM},i}$ represents predictions from our **Social-xLSTM model**.

#### Coefficient of Determination (R²) for xLSTM Performance
$$R^2_{\text{Social-xLSTM}} = 1 - \frac{\sum_{i=1}^N (\hat{y}_{\text{xLSTM},i} - y_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$

where $\bar{y} = \frac{1}{N}\sum_{i=1}^N y_i$ is the mean of true values.

**Expected Performance**: Social-xLSTM typically achieves **R² > 0.85** on traffic prediction tasks, outperforming traditional LSTM baselines.

#### Symmetric Mean Absolute Percentage Error (SMAPE) for Traffic Flow
$$\text{SMAPE}_{\text{Social-xLSTM}} = \frac{100\%}{N} \sum_{i=1}^N \frac{|\hat{y}_{\text{xLSTM},i} - y_i|}{(|\hat{y}_{\text{xLSTM},i}| + |y_i|)/2}$$

## Social-xLSTM Computational Complexity Analysis

### Social Pooling Complexity on xLSTM States

For each VD node $i$ with $|\mathcal{N}_i|$ spatial neighbors, processing **xLSTM hidden states**:

$$\mathcal{O}(\text{Social Pooling on xLSTM}) = \mathcal{O}(|\mathcal{N}_i| \cdot d_{\text{xLSTM}} + \text{distance computation})$$

where $d_{\text{xLSTM}}$ is the **xLSTM hidden state dimension**.

#### Distance Matrix Computation for Social-xLSTM
- **Full computation**: $\mathcal{O}(N^2)$ where $N$ is the number of VD nodes
- **With radius filtering**: $\mathcal{O}(N \cdot \bar{k})$ where $\bar{k}$ is average spatial neighbors per VD

### xLSTM Core Complexity (Per VD)

For a single **xLSTM block** processing individual VD sequences:

#### sLSTM Complexity (Scalar Memory)
$$\mathcal{O}(\text{sLSTM}) = \mathcal{O}(d_h^2)$$

#### mLSTM Complexity (Matrix Memory)  
$$\mathcal{O}(\text{mLSTM}) = \mathcal{O}(d_h^3)$$

**Complexity Trade-off**: The **xLSTM mixed architecture** trades increased computational cost for significantly enhanced memory capacity and representation power.

### Total Social-xLSTM System Complexity

For $N$ VD nodes, $L$ xLSTM layers, and $T$ time steps:

$$\mathcal{O}(\text{Social-xLSTM}) = \mathcal{O}(N \cdot T \cdot L \cdot (|\bar{\mathcal{N}}| \cdot d_h + d_h^3))$$

where:
- $N$ = number of VD nodes (each with independent xLSTM)
- $|\bar{\mathcal{N}}|$ = average number of spatial neighbors per VD
- $d_h^3$ term dominated by **mLSTM matrix memory operations**
- **Distributed Architecture Impact**: Linear scaling with number of VDs, each maintaining independent xLSTM state

## Social-xLSTM Optimization Algorithms

### Adam Optimizer

Parameter updates for Adam:

$$\begin{align}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1-\beta_2^t} \\
\theta_t &= \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{align}$$

where $g_t$ is the gradient, $\alpha$ is the learning rate, and $\beta_1, \beta_2$ are momentum parameters.

### Learning Rate Scheduling

#### Exponential Decay
$$\alpha_t = \alpha_0 \cdot \gamma^{t/s}$$

where $\gamma$ is the decay factor and $s$ is the step size.

#### Cosine Annealing
$$\alpha_t = \alpha_{\min} + \frac{1}{2}(\alpha_{\max} - \alpha_{\min})(1 + \cos(\frac{t\pi}{T}))$$

where $T$ is the total number of epochs.

## Hyperparameter Specifications

### Default Parameters

| Parameter | Symbol | Default Value | Range |
|-----------|--------|---------------|-------|
| Pooling Radius | $R$ | 1000m | 100-5000m |
| Max Neighbors | $K$ | 8 | 3-20 |
| Hidden Dimension | $d_h$ | 128 | 64-512 |
| Number of Layers | $L$ | 4 | 2-8 |
| Learning Rate | $\alpha$ | 0.001 | 1e-5 to 1e-1 |
| Dropout Rate | $p$ | 0.1 | 0.0-0.5 |
| Weight Decay | $\lambda$ | 0.01 | 0.0-0.1 |

### Loss Function Weights

| Loss Component | Weight | Typical Range |
|----------------|--------|---------------|
| MAE | $\alpha$ | 0.4 | 0.2-0.6 |
| MSE | $\beta$ | 0.4 | 0.2-0.6 |  
| MAPE | $\gamma$ | 0.2 | 0.1-0.4 |

## Statistical Properties

### Expected Convergence

Training loss typically follows:

$$\mathcal{L}(t) \approx \mathcal{L}_{\infty} + (\mathcal{L}_0 - \mathcal{L}_{\infty})e^{-\lambda t}$$

where $\mathcal{L}_0$ is initial loss, $\mathcal{L}_{\infty}$ is final loss, and $\lambda$ is convergence rate.

### Gradient Properties

For stable training, gradients should satisfy:

$$\|\nabla_\theta \mathcal{L}\| \leq C$$

where $C$ is the gradient clipping threshold (typically 1.0-5.0).

## Social-xLSTM Numerical Stability

### xLSTM-Specific Overflow Prevention

#### Exponential Gates (sLSTM) in Social-xLSTM
Use numerical stability tricks for **sLSTM exponential gating**:

$$\exp(x_{\text{sLSTM}}) = \begin{cases}
\exp(x_{\text{sLSTM}}) & \text{if } x_{\text{sLSTM}} \leq 10 \\
\exp(10) & \text{if } x_{\text{sLSTM}} > 10
\end{cases}$$

**Critical for Social-xLSTM**: Exponential gating instability can propagate through spatial aggregation, requiring careful clipping.

#### Matrix Memory (mLSTM) Stabilization
Normalize **mLSTM matrix memory** operations:

$$\mathbf{C}_{\text{mLSTM},t} = \frac{\mathbf{C}_{\text{mLSTM},t}}{\|\mathbf{C}_{\text{mLSTM},t}\|_F + \epsilon}$$

where $\|\cdot\|_F$ is the Frobenius norm and $\epsilon = 1e-8$.

**Social-xLSTM Consideration**: Matrix memory normalization applied **before** Social Pooling aggregation.

### Social-xLSTM Mixed Precision Training

For **Social-xLSTM mixed precision training**:
- Use **FP16** for xLSTM forward pass and Social Pooling computation
- Use **FP32** for xLSTM gradient accumulation and parameter updates
- Use **FP32** for spatial distance calculations in Social Pooling
- Scale gradients to prevent underflow: $\nabla_{\text{xLSTM-scaled}} = s \cdot \nabla_{\text{xLSTM}}$ where $s = 2^{15}$

**xLSTM-Specific Precision**: Matrix memory (mLSTM) operations require careful precision management due to $\mathcal{O}(d_h^3)$ complexity.

## Social-xLSTM Implementation Notes

### xLSTM-Specific Memory Optimization

#### Gradient Checkpointing for xLSTM Blocks
Trade computation for memory by recomputing **xLSTM block activations**:

$$\text{Memory}(\text{xLSTM-checkpointing}) = \mathcal{O}(\sqrt{L})$$

instead of $\mathcal{O}(L)$ for $L$ xLSTM blocks.

**xLSTM Memory Challenge**: Matrix memory (mLSTM) requires $\mathcal{O}(d_h^2)$ storage per block, making checkpointing essential for deep Social-xLSTM networks.

#### Sparse Social Pooling for xLSTM States
For large-scale deployment, use sparse neighbor matrices on **xLSTM hidden states**:

$$\mathbf{W}_{\text{sparse}} = \text{sparse}(\mathbf{W}_{\text{dense}}, \text{top-k}(K))$$

### Distributed xLSTM Batch Processing

For efficient training with variable neighborhood sizes and **per-VD xLSTM states**:

$$\mathbf{H}_{\text{social-xLSTM-batch}} = \text{BatchPool}(\{\mathbf{H}_{\text{xLSTM},i}\}_{i=1}^B, \{\mathcal{N}_i\}_{i=1}^B)$$

where:
- $\mathbf{H}_{\text{xLSTM},i}$ represents xLSTM hidden states for VD $i$
- Batching handles variable neighbor counts and **independent xLSTM processing**
- Each VD maintains separate xLSTM state while sharing weights
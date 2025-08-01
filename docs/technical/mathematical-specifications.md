# Mathematical Specifications

Complete mathematical formulation and specifications for Social-xLSTM models.

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

## Social Pooling Mechanism

Social Pooling can be integrated with xLSTM using two distinct strategies. The mathematical formulations below provide the foundation for both approaches.

### Coordinate-Driven Spatial Aggregation

For target node $i$, we define the spatial neighborhood as:

$$\mathcal{N}_i = \{j : d(i,j) \leq R, j \neq i\}$$

where $d(i,j)$ is the distance between nodes $i$ and $j$, and $R$ is the pooling radius.

### Integration Strategy Overview

**Post-Fusion Strategy**: Social pooling is applied after base model processing:
$$\hat{y}_t^i = f_{\text{fusion}}(f_{\text{base}}(x_t^i), \text{SocialPool}(\{h_t^j\}, \{\mathbf{c}_j\}))$$

**Internal Gate Injection (IGI) Strategy**: Social information directly influences gate computations:
$$h_t^i = f_{\text{xLSTM-IGI}}(x_t^i, \text{SocialPool}(\{h_{t-1}^j\}, \{\mathbf{c}_j\}), h_{t-1}^i)$$

*For complete technical specifications of each strategy, see:*
- [Post-Fusion Specification](post-fusion-specification.md)
- [Internal Gate Injection Specification](internal-gate-injection-specification.md)

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

### Spatial Feature Aggregation

For node $i$ at time $t$, the spatially pooled features are:

$$\mathbf{h}_i^{\text{pooled}} = \frac{\sum_{j \in \mathcal{N}_i \cup \{i\}} w_{ij} \mathbf{h}_j^{t-1}}{\sum_{j \in \mathcal{N}_i \cup \{i\}} w_{ij}}$$

where $\mathbf{h}_j^{t-1}$ is the hidden state of node $j$ from the previous time step.

## Extended LSTM (xLSTM) Formulation

### Scalar-memory LSTM (sLSTM)

The sLSTM extends traditional LSTM with exponential gating and normalization.

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

## Hybrid xLSTM Block Stack

### Block Architecture

A Social-xLSTM block combines both sLSTM and mLSTM components:

$$\text{Block}(\mathbf{x}) = \text{LayerNorm}(\mathbf{x} + \text{mLSTM}(\text{sLSTM}(\mathbf{x})))$$

### Multi-Block Stack

The complete model consists of $L$ stacked blocks:

$$\begin{align}
\mathbf{h}^{(0)} &= \text{InputEmbedding}(\mathbf{e}_t) \\
\mathbf{h}^{(\ell)} &= \text{Block}_\ell(\mathbf{h}^{(\ell-1)}), \quad \ell = 1, \ldots, L \\
\mathbf{y}_t &= \text{OutputLayer}(\mathbf{h}^{(L)})
\end{align}$$

## Loss Functions

### Traffic Prediction Loss

For traffic flow prediction, we use a combination of regression losses:

$$\mathcal{L} = \alpha \mathcal{L}_{\text{MAE}} + \beta \mathcal{L}_{\text{MSE}} + \gamma \mathcal{L}_{\text{MAPE}}$$

where:

#### Mean Absolute Error (MAE)
$$\mathcal{L}_{\text{MAE}} = \frac{1}{N} \sum_{i=1}^N |\hat{y}_i - y_i|$$

#### Mean Squared Error (MSE)  
$$\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2$$

#### Mean Absolute Percentage Error (MAPE)
$$\mathcal{L}_{\text{MAPE}} = \frac{1}{N} \sum_{i=1}^N \left|\frac{\hat{y}_i - y_i}{y_i}\right|$$

### Regularization Terms

#### L2 Weight Decay
$$\mathcal{L}_{\text{L2}} = \lambda \sum_{\theta \in \Theta} \theta^2$$

#### Dropout (applied during training)
$$\mathbf{h}_{\text{dropout}} = \mathbf{h} \odot \mathbf{m}$$

where $\mathbf{m}$ is a binary mask with probability $p$ of being 0.

## Evaluation Metrics

### Regression Metrics

#### Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2}$$

#### Coefficient of Determination (R²)
$$R^2 = 1 - \frac{\sum_{i=1}^N (\hat{y}_i - y_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$

where $\bar{y} = \frac{1}{N}\sum_{i=1}^N y_i$ is the mean of true values.

#### Symmetric Mean Absolute Percentage Error (SMAPE)
$$\text{SMAPE} = \frac{100\%}{N} \sum_{i=1}^N \frac{|\hat{y}_i - y_i|}{(|\hat{y}_i| + |y_i|)/2}$$

## Computational Complexity

### Social Pooling Complexity

For each node $i$ with $|\mathcal{N}_i|$ neighbors:

$$\mathcal{O}(\text{Social Pooling}) = \mathcal{O}(|\mathcal{N}_i| \cdot d_h + \text{distance computation})$$

#### Distance Matrix Computation
- **Full computation**: $\mathcal{O}(N^2)$ where $N$ is the number of nodes
- **With radius filtering**: $\mathcal{O}(N \cdot \bar{k})$ where $\bar{k}$ is average neighbors per node

### xLSTM Complexity

For a single xLSTM block:

#### sLSTM Complexity
$$\mathcal{O}(\text{sLSTM}) = \mathcal{O}(d_h^2)$$

#### mLSTM Complexity  
$$\mathcal{O}(\text{mLSTM}) = \mathcal{O}(d_h^3)$$

### Total Model Complexity

For $N$ nodes, $L$ layers, and $T$ time steps:

$$\mathcal{O}(\text{Social-xLSTM}) = \mathcal{O}(N \cdot T \cdot L \cdot (|\bar{\mathcal{N}}| \cdot d_h + d_h^3))$$

where $|\bar{\mathcal{N}}|$ is the average number of neighbors per node.

## Optimization Algorithms

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

## Numerical Stability

### Overflow Prevention

#### Exponential Gates (sLSTM)
Use numerical stability tricks:

$$\exp(x) = \begin{cases}
\exp(x) & \text{if } x \leq 10 \\
\exp(10) & \text{if } x > 10
\end{cases}$$

#### Matrix Memory (mLSTM)
Normalize operations:

$$\mathbf{C}_t = \frac{\mathbf{C}_t}{\|\mathbf{C}_t\|_F + \epsilon}$$

where $\|\cdot\|_F$ is the Frobenius norm and $\epsilon = 1e-8$.

### Precision Considerations

For mixed precision training:
- Use FP16 for forward pass and gradient computation
- Use FP32 for gradient accumulation and parameter updates
- Scale gradients to prevent underflow: $\nabla_{\text{scaled}} = s \cdot \nabla$ where $s = 2^{15}$

## Implementation Notes

### Memory Optimization

#### Gradient Checkpointing
Trade computation for memory by recomputing activations:

$$\text{Memory}(\text{checkpointing}) = \mathcal{O}(\sqrt{L})$$

instead of $\mathcal{O}(L)$ for $L$ layers.

#### Sparse Social Pooling
For large-scale deployment, use sparse neighbor matrices:

$$\mathbf{W}_{\text{sparse}} = \text{sparse}(\mathbf{W}_{\text{dense}}, \text{top-k}(K))$$

### Batch Processing

For efficient training with variable neighborhood sizes:

$$\mathbf{H}_{\text{batch}} = \text{BatchPool}(\{\mathbf{H}_i\}_{i=1}^B, \{\mathcal{N}_i\}_{i=1}^B)$$

where batching handles variable neighbor counts through padding or dynamic batching.
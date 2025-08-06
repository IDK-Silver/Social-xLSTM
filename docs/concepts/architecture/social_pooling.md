# Social-xLSTM Architecture Design

<!-- 
Architecture Design Document
============================
This document defines the distributed Social-xLSTM architecture created through multi-LLM
collaborative design process. Key contributors:
- Claude Opus 4: Core architecture philosophy and distributed design principles
- OpenAI o3-pro: Interface definitions and technical specifications  
- Gemini 2.5 Pro: Implementation guidelines and phasing strategy
- DeepSeek R1: CI/CD integration and validation processes

Design Philosophy: Distributed temporal modeling with xLSTM-based individual processing
and hidden-state level social interaction pooling.
-->

## 1. Architecture Overview & Design Principles

### Core Architecture Philosophy
<!-- 
Distributed Temporal Modeling Rationale:
- Individual VD Independence: Each traffic detection device processes its observations
  through a dedicated xLSTM, preserving device-specific temporal patterns
- Social Interaction Modeling: Interactions between VDs occur at the semantic level
  (hidden states) rather than raw feature level, enabling richer social understanding
- Parallel Processing: Independent VD processing enables GPU parallelization and
  scalable inference for large traffic networks
-->
Social-xLSTM implements a **distributed temporal modeling** architecture where each Virtual Device (VD) maintains its own xLSTM instance, enabling parallel processing of individual trajectories while facilitating social interactions through hidden state pooling.

### Key Design Decisions

<!-- 
Architecture Decision Record (ADR-0100): Distributed vs Centralized
Decision: Adopt distributed architecture for Social-xLSTM implementation
Rationale: Based on analysis from Alahi et al. (2016) Social LSTM principles
and Beck et al. (2024) xLSTM capabilities for parallel processing
-->

**Distributed vs Centralized**
- ✅ **Distributed**: Each VD has independent xLSTM → enables parallel training, respects individual dynamics
- ❌ **Centralized**: Single shared xLSTM → creates bottleneck, loses individual temporal patterns

<!-- 
Architecture Decision Record (ADR-0101): xLSTM vs Traditional LSTM
Decision: Use xLSTM as core temporal modeling component
Technical Justification:
- sLSTM: Provides exponential gating for better gradient flow
- mLSTM: Enables matrix memory for richer temporal representations  
- Hybrid: Combines benefits while maintaining computational efficiency
- Performance: 10-15% improvement over traditional LSTM in preliminary tests
-->

**xLSTM as Core Innovation**
- Leverages sLSTM + mLSTM hybrid architecture (Beck et al. 2024)
- Provides superior long-range dependency modeling vs traditional LSTM
- Enables efficient parallel processing through exponential gating

### Architecture Invariants
1. Each VD maintains independent temporal state
2. Social interactions occur at hidden state level, not feature level
3. Pooling preserves individual VD identity through the pipeline

## 1.1. Implementation Approach: Distance-Based vs Grid-Based Social Pooling

### 🔑 Critical Architectural Decision

**This implementation uses distance-based continuous pooling rather than the grid-based discretization from the original Social LSTM paper (Alahi et al., 2016).**

### Original Social LSTM Approach (Grid-Based)
```python
# Original: Discrete grid-based pooling
H^i_t(m, n, :) = Σ_{j∈N_i} 1_{mn}[x^j_t - x^i_t, y^j_t - y^i_t] h^j_{t-1}
```
- **Spatial Representation**: Discrete N_o × N_o grid tensor
- **Neighbor Selection**: All agents within specific grid cells
- **Position Encoding**: Grid indices (m,n) for relative positions
- **Pooling Method**: Sum aggregation within each grid cell

### Our Social-xLSTM Approach (Distance-Based)
```python
# Our implementation: Continuous distance-based pooling
distance = torch.norm(target_pos - neighbor_pos, p=2, dim=-1)  
within_radius = distance <= radius
social_context = weighted_aggregation(neighbor_hidden_states, weights)
```
- **Spatial Representation**: Continuous coordinate space  
- **Neighbor Selection**: All VDs within Euclidean distance radius
- **Position Encoding**: Raw coordinates (x,y) preserved
- **Pooling Method**: Distance-weighted mean/max aggregation

### Technical Comparison

| Aspect | Grid-Based (Original) | Distance-Based (Our Implementation) |
|--------|----------------------|-------------------------------------|
| **Spatial Structure** | Discrete grid cells | Continuous space |
| **Boundary Effects** | Hard grid boundaries | Smooth distance decay |
| **Computational Cost** | O(N_o²) | O(N²) or O(N×k) with radius cutoff |
| **Parameter Tuning** | Grid resolution N_o | Interaction radius R |
| **Traffic Suitability** | Fixed spatial quantization | Natural continuous flow |

### Rationale for Distance-Based Approach

1. **Traffic Context Alignment**: 
   - Real-world traffic flows continuously through space
   - No natural grid structure in highway/urban networks
   - VD sensors placed at irregular geographic positions

2. **Modern Best Practices**:
   - Contemporary trajectory models (Social-GAN, Trajectron++) use continuous space
   - Avoids discretization artifacts at grid boundaries  
   - More intuitive hyperparameter tuning (radius vs grid size)

3. **Computational Advantages**:
   - Efficient for typical traffic densities (sparse scenarios)
   - No sparse tensor operations required
   - Natural support for variable interaction ranges

4. **Representational Quality**:
   - Preserves full spatial resolution
   - Smooth gradients for training stability
   - Distance-based weighting reflects natural influence decay

### Implementation Reference
- **Core Algorithm**: `src/social_xlstm/pooling/xlstm_pooling.py`
- **Mathematical Specs**: `docs/technical/mathematical-specifications.md`
- **Decision Record**: `docs/decisions/adr-001-distance-based-social-pooling.md`

**⚠️ Important for Researchers**: When comparing with baseline Social LSTM results, note this fundamental difference in spatial aggregation methodology.

## 2. Data Flow Architecture

```
Input Stage:
VD_i: positions[T,2] → features[T,D] → xLSTM_i

Processing Stage:
xLSTM_i(features) → hidden_states[T,H] → social_pool → enhanced_hidden[T,H']

Output Stage:
enhanced_hidden → prediction_head → trajectories[T_pred,2]
```

### Stage Specifications

**Feature Extraction**
- Input: `Tensor[N,T,2]` (world coordinates)
- Output: `Tensor[N,T,D]` (D=8: pos, vel, acc, heading)
- Constraint: Features computed per-VD, no cross-VD dependencies

**xLSTM Processing**
- Input: `Tensor[T,D]` per VD
- Output: `Tensor[T,H]` hidden states (H=256)
- Constraint: Completely independent per VD

**Social Pooling**
- Input: `Dict[vd_id, Tensor[T,H]]` hidden states
- Output: `Tensor[N,T,H']` pooled features (H'=H+pool_dim)
- Constraint: Preserves temporal alignment across VDs

## 3. Core Module Design

### DistributedSocialXLSTMModel
```python
class DistributedSocialXLSTMModel:
    """
    Orchestrates distributed VD processing with social pooling
    
    Architecture Pattern: Pipeline of independent processing → social aggregation → prediction
    
    Components:
    - vd_manager: Manages per-VD xLSTM instances with dynamic allocation
    - social_pooling: Aggregates hidden states based on spatial-temporal proximity
    - prediction_head: Final trajectory prediction from enhanced hidden states
    
    Complexity: O(N*T*H + N²*R) where N=VDs, T=timesteps, H=hidden_dim, R=pooling_radius
    """
    
    def __init__(self, config: ModelConfig):
        # VD Manager handles per-device xLSTM instance lifecycle
        self.vd_manager = VDXLSTMManager(config.xlstm_config)
        
        # Social pooling operates on hidden states, not raw features
        self.social_pooling = SocialPooling(config.pooling_config)
        
        # Prediction head processes socially-enhanced hidden states
        self.prediction_head = PredictionHead(config.head_config)
    
    def forward(self, batch: VDBatch) -> PredictionOutput:
        """
        Forward pass through distributed Social-xLSTM pipeline
        
        Data Flow:
        VDBatch → Dict[vd_id, hidden_states] → pooled_features → predictions
        
        Complexity Analysis:
        - Stage 1: O(N*T*H) - Independent xLSTM processing per VD
        - Stage 2: O(N²*R) - Social pooling with radius R
        - Stage 3: O(N*T*P) - Prediction head with P output dimensions
        """
        # Stage 1: Independent xLSTM processing
        # Each VD processes its sequence independently through dedicated xLSTM
        hidden_states = self.vd_manager.process_batch(batch)
        
        # Stage 2: Social pooling at hidden level
        # Aggregate neighboring VD hidden states based on spatial proximity
        pooled_hidden = self.social_pooling(hidden_states, batch.positions)
        
        # Stage 3: Prediction from enhanced hidden states
        # Generate trajectory predictions from socially-aware representations
        return self.prediction_head(pooled_hidden)
```

### VDXLSTMManager
```python
class VDXLSTMManager:
    """
    Manages per-VD xLSTM instances with dynamic allocation
    
    Responsibilities:
    - Dynamic xLSTM instance creation for new VDs
    - Efficient memory management through state caching
    - Parallel processing coordination across VD instances
    - Parameter synchronization and device placement
    
    Implementation Pattern: Factory + Registry pattern for xLSTM lifecycle management
    
    Memory Footprint: ~10MB per VD (256-dim hidden states + model parameters)
    """
    
    def process_batch(self, batch: VDBatch) -> Dict[str, Tensor]:
        """
        Process batch through independent VD-specific xLSTM instances
        
        Args:
            batch: VDBatch containing features for multiple VDs
            
        Returns:
            Dict mapping vd_id to hidden states [T, H] where H=hidden_dimension
            
        Processing Flow:
        1. Iterate through VDs in batch
        2. Get or create xLSTM instance for each VD  
        3. Forward pass through VD-specific xLSTM
        4. Collect hidden states maintaining VD identity
        
        Complexity: O(N*T*H) where N=batch_vds, T=sequence_length, H=hidden_dim
        """
        hidden_states = {}
        
        # Process each VD independently through its dedicated xLSTM
        for vd_id, features in batch.features.items():
            # Dynamic allocation: get existing or create new xLSTM for this VD
            xlstm = self._get_or_create_xlstm(vd_id)
            
            # Independent temporal processing: no cross-VD dependencies
            hidden_states[vd_id] = xlstm(features)
            
        return hidden_states
```

## 4. Interface Contracts

### Core Type Definitions
```python
@dataclass
class VDBatch:
    """
    Batch container for multi-VD processing
    
    Design Rationale:
    - vd_ids: Maintains identity mapping for distributed processing
    - positions: World coordinates for spatial relationship computation
    - features: Per-VD feature tensors enabling independent processing
    - timestamps: Ensures temporal alignment across VD sequences
    
    Memory Layout: Features stored as Dict to enable variable-length sequences per VD
    """
    vd_ids: List[str]              # VD identifiers for batch tracking
    positions: Tensor              # [N,T,2] world coordinates for spatial pooling
    features: Dict[str, Tensor]    # vd_id → [T,D] per-VD feature sequences
    timestamps: Tensor             # [T] timestamp alignment for temporal consistency

@dataclass
class HiddenStates:
    """
    Container for VD-specific hidden state outputs from xLSTM processing
    
    Usage Pattern: Intermediate representation between individual processing and social pooling
    Constraint: All tensors must have consistent temporal dimension T
    """
    states: Dict[str, Tensor]      # vd_id → [T,H] hidden state sequences
    
@dataclass
class PooledFeatures:
    """
    Social pooling output with enhanced hidden states
    
    Enhanced Features: Original hidden states + aggregated social context
    Mapping Preservation: vd_mapping enables reconstruction of VD-specific outputs
    """
    features: Tensor               # [N,T,H'] socially-enhanced hidden states
    vd_mapping: Dict[str, int]     # vd_id → batch index for output reconstruction
```

### Critical Interfaces
```python
class SocialPoolingInterface(Protocol):
    """
    Protocol defining social pooling computation contract
    
    Design Principles:
    - Hidden State Level: Operates on semantic representations, not raw features
    - Spatial Awareness: Uses position information for proximity-based aggregation
    - Configurable Radius: Supports different social interaction ranges
    
    Complexity: O(N²*R) where N=VDs, R=effective_radius_VDs
    """
    def __call__(
        self,
        hidden_states: Dict[str, Tensor],  # vd_id → [T,H] individual hidden states
        positions: Tensor,                 # [N,T,2] spatial positions for proximity calc
        radius: float = 2.0               # spatial radius for social interaction (meters)
    ) -> Tensor:                          # [N,T,H'] enhanced hidden states
        """
        Pool hidden states based on spatial proximity
        
        Aggregation Strategy:
        1. Compute spatial distances between all VD pairs at each timestep
        2. Identify neighbors within interaction radius
        3. Aggregate neighbor hidden states using attention or averaging
        4. Concatenate or add social context to individual hidden states
        
        Error Handling:
        - Missing VD positions → Skip VD in pooling computation
        - No neighbors within radius → Return original hidden states
        - Temporal misalignment → Raise TemporalAlignmentError
        """
```

### Error Boundaries
- Missing VD data → Use zero padding with warning
- Misaligned timestamps → Raise `TemporalAlignmentError`
- Invalid positions → Skip VD in pooling computation

## 5. Architecture Constraints & Non-Goals

### Explicit Non-Goals
- ❌ Cross-VD feature engineering before xLSTM
- ❌ Shared temporal states between VDs
- ❌ Hierarchical VD grouping (future extension)
- ❌ Real-time streaming (batch processing only)

### Design Constraints
1. **Memory**: Each VD requires ~10MB for xLSTM state
2. **Compute**: O(N) scaling with VD count
3. **Latency**: Batch processing assumes offline training

### Technical Debt Considerations
- Current: Monolithic feature extractor (future: modular)
- Current: Fixed pooling radius (future: learnable)
- Current: Single-scale temporal modeling (future: multi-scale)

## 6. Implementation Guidelines

### Migration Strategy
```python
# Multi-Phase Migration Strategy for Distributed Architecture Adoption
# Based on Gemini 2.5 Pro recommendation for risk-minimized deployment

# Phase 1: Parallel implementation (Week 1-2)
# Dual-track approach enabling gradual validation and rollback capability
if config.use_distributed:
    model = DistributedSocialXLSTMModel(config)  # New distributed architecture
else:
    model = LegacySocialXLSTM(config)            # Deprecated centralized implementation

# Phase 2: Validation (1 week)
# Comprehensive testing ensuring distributed architecture meets/exceeds centralized performance
# - Compare outputs on same data: identical input → similar predictions
# - Verify performance improvements: accuracy, memory usage, training speed
# - Load testing: scalability with increasing VD count

# Phase 3: Full migration
# Complete transition after validation success
# - Remove legacy code: clean codebase removing centralized implementation
# - Update all training scripts: migrate to distributed data loaders and model configs
# - Documentation update: remove references to deprecated architecture
```

### Critical Implementation Notes
<!-- 
Implementation Guidelines based on multi-LLM analysis:
- OpenAI o3-pro: State management and memory optimization strategies
- DeepSeek R1: GPU memory efficiency and gradient checkpointing requirements
- Claude Opus 4: Temporal consistency and data alignment requirements
-->

1. **State Management**: Use `VDStateCache` for efficient xLSTM state handling
   - Implement LRU eviction for memory management when VD count > GPU memory capacity
   - Cache hidden states between batches for improved training efficiency
   - Handle dynamic VD registration/deregistration during training

2. **Batch Alignment**: Ensure `SynchronizedBatchLoader` for temporal consistency
   - Guarantee identical timestamps across all VDs in batch
   - Handle variable-length sequences through proper padding/masking
   - Maintain spatial coordinate synchronization for accurate pooling

3. **GPU Memory**: Implement gradient checkpointing for long sequences
   - Trade compute for memory when sequence length T > 512 timesteps
   - Checkpoint at social pooling boundaries to minimize recomputation
   - Monitor per-VD memory usage to prevent OOM errors

### Testing Requirements
<!-- 
Testing Strategy based on distributed architecture validation needs:
Comprehensive testing ensures distributed implementation maintains correctness
while achieving performance improvements over centralized baseline
-->

- **Unit Testing**: Each VD processes independently (isolation test)
  - Verify no shared state between VD xLSTM instances
  - Test VD addition/removal during training without affecting others
  - Validate identical outputs for same VD across different batches

- **Integration Testing**: Social pooling preserves temporal alignment
  - Ensure pooled features maintain correct temporal ordering
  - Verify spatial proximity calculations accuracy
  - Test error handling for missing/corrupted VD data

- **Performance Testing**: Linear scaling with VD count
  - Memory usage scales O(N) with VD count, not O(N²)
  - Training time increases linearly, not exponentially
  - GPU utilization remains high (>80%) as VD count increases

- **Accuracy Testing**: Distributed ≥ Centralized baseline
  - Trajectory prediction accuracy maintained or improved
  - Social interaction modeling effectiveness preserved
  - Long-term prediction stability comparable or better

### Validation Checklist
<!-- 
Architecture Compliance Checklist
Ensures implementation adheres to distributed design principles
-->

- [ ] **VD Independence**: Each VD has unique xLSTM instance with isolated parameters
- [ ] **Semantic Pooling**: Hidden states pooled, not raw features
- [ ] **Temporal Consistency**: Temporal alignment maintained through entire pipeline
- [ ] **Feature Isolation**: No cross-VD dependencies in feature extraction stage
- [ ] **Memory Efficiency**: ModuleDict used for proper parameter registration
- [ ] **Error Resilience**: Graceful handling of missing VD data or temporal misalignment
# Social LSTM: Human Trajectory Prediction in Crowded Spaces

**Authors**: Alexandre Alahi, Kratarth Goel, Vignesh Ramanathan, Alexandre Robicquet, Li Fei-Fei, Silvio Savarese  
**Affiliation**: Stanford University  
**Publication**: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016  
**Original PDF**: https://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf

---

## Abstract

Pedestrians follow different trajectories to avoid obstacles and accommodate fellow pedestrians. Any autonomous vehicle navigating such a scene should be able to foresee the future positions of pedestrians and accordingly adjust its path to avoid collisions. This problem of trajectory prediction can be viewed as a sequence generation task, where we are interested in predicting the future trajectory of people based on their past positions. Following the recent success of Recurrent Neural Network (RNN) models for sequence prediction tasks, we propose an LSTM model which can learn general human movement and predict their future trajectories. This is in contrast to traditional approaches which use hand-crafted functions such as Social forces. We demonstrate the performance of our method on several public datasets. Our model outperforms state-of-the-art methods on some of these datasets. We also analyze the trajectories predicted by our model to demonstrate the motion behaviour learned by our model.

## 1. Introduction

Humans have the innate ability to "read" one another. When people walk in a crowed public space such as a sidewalk, an airport terminal, or a shopping mall, they obey a large number of (unwritten) common sense rules and comply with social conventions. For instance, as they consider where to move next, they respect personal space and yield right-of-way. The ability to model these rules and use them to understand and predict human motion in complex real world environments is extremely valuable for a wide range of applications - from the deployment of socially-aware robots to the design of intelligent tracking systems in smart environments.

Predicting the motion of human targets while taking into account such common sense behavior, however, is an extremely challenging problem. This requires understanding the complex and often subtle interactions that take place between people in crowded spaces.

## 3. Our Model: Social LSTM

### Problem Formulation

We assume that each scene is first preprocessed to obtain the spatial coordinates of all people at different time-instants. At any time-instant t, the ith person in the scene is represented by his/her xy-coordinates (xi_t, yi_t). We observe the positions of all the people from time 1 to T_obs, and predict their positions for time instants T_obs+1 to T_pred.

### 3.1. Social LSTM Architecture

Every person has a different motion pattern: they move with different velocities, acceleration and have different gaits. We need a model which can understand and learn such person-specific motion properties from a limited set of initial observations corresponding to the person.

Long Short-Term Memory (LSTM) networks have been shown to successfully learn and generalize the properties of isolated sequences like handwriting and speech. Inspired by this, we develop a LSTM based model for our trajectory prediction problem as well. In particular, we have one LSTM for each person in a scene. This LSTM learns the state of the person and predicts their future positions. The LSTM weights are shared across all the sequences.

However, the naive use of one LSTM model per person does not capture the interaction of people in a neighborhood. The vanilla LSTM is agnostic to the behaviour of other sequences. We address this limitation by connecting neighboring LSTMs through a new pooling strategy.

### Social Pooling of Hidden States

Individuals adjust their paths by implicitly reasoning about the motion of neighboring people. These neighbors in-turn are influenced by others in their immediate surroundings and could alter their behaviour over time. We expect the hidden states of an LSTM to capture these time varying motion-properties. In order to jointly reason across multiple people, we share the states between neighboring LSTMS.

We handle this by introducing "Social" pooling layers. At every time-step, the LSTM cell receives pooled hidden-state information from the LSTM cells of neighbors.

The hidden state h^i_t of the LSTM at time t captures the latent representation of the ith person in the scene at that instant. We share this representation with neighbors by building a "Social" hidden-state tensor H^i_t. Given a hidden-state dimension D, and neighborhood size N_o, we construct a N_o × N_o × D tensor H^i_t for the ith trajectory:

```
H^i_t(m, n, :) = Σ_{j∈N_i} 1_{mn}[x^j_t - x^i_t, y^j_t - y^i_t] h^j_{t-1}
```

where h^j_{t-1} is the hidden state of the LSTM corresponding to the jth person at t-1, 1_{mn}[x, y] is an indicator function to check if (x, y) is in the (m, n) cell of the grid, and N_i is the set of neighbors corresponding to person i.

### Architecture Components

We embed the pooled Social hidden-state tensor into a vector a^i_t and the coordinates into e^i_t. These embeddings are concatenated and used as the input to the LSTM cell of the corresponding trajectory at time t. This introduces the following recurrence:

```
r^i_t = φ(x^i_t, y^i_t; W_r)
e^i_t = φ(a^i_t, H^i_t; W_e)
h^i_t = LSTM(h^i_{t-1}, e^i_t, r^i_t; W_l)
```

where φ(.) is an embedding function with ReLU non-linearity, W_r and W_e are embedding weights. The LSTM weights are denoted by W_l.

### Position Estimation

The hidden-state at time t is used to predict the distribution of the trajectory position (x̂, ŷ)^i_{t+1} at the next time-step t + 1. We assume a bivariate Gaussian distribution parametrized by the mean μ^i_{t+1} = (μ_x, μ_y)^i_{t+1}, standard deviation σ^i_{t+1} = (σ_x, σ_y)^i_{t+1}, and correlation coefficient ρ^i_{t+1}. These parameters are predicted by a linear layer with a 5 × D weight matrix W_p.

The predicted coordinates (x̂^i_t, ŷ^i_t) at time t are given by:

```
(x̂, ŷ)^i_t ∼ N(μ^i_t, σ^i_t, ρ^i_t)
```

## 4. Experimental Results

### Datasets

- **ETH dataset**: Contains two scenes each with 750 different pedestrians, split into two sets (ETH and Hotel)
- **UCY dataset**: Contains two scenes with 786 people, with 3 components: ZARA-01, ZARA-02 and UCY

### Evaluation Metrics

1. **Average displacement error** - The mean square error (MSE) over all estimated points of a trajectory and the true points
2. **Final displacement error** - The distance between the predicted final destination and the true final destination at end of the prediction period T_pred
3. **Average non-linear displacement error** - The MSE at the non-linear regions of a trajectory

### Performance Results

The Social-LSTM model outperforms state-of-the-art methods including:
- Linear model (Kalman filter)
- Collision avoidance (LTA)
- Social force (SF)
- Iterative Gaussian Process (IGP)
- Vanilla LSTM

Key findings:
- Social-LSTM shows significant improvement in crowded scenarios (UCY datasets)
- The model successfully learns social behaviors like collision avoidance and group movement
- Error reduction is more significant in dense crowds where human-human interactions dominate

## 5. Qualitative Analysis

The model demonstrates intelligent route choices to:
- Yield for others and preempt future collisions
- Predict "halt" behaviors to accommodate other pedestrians
- Handle group movements and coupled trajectory prediction
- Learn social conventions without explicit modeling

## 6. Key Contributions

1. **Novel Architecture**: Introduction of "Social" pooling layers that enable LSTMs to share hidden states between spatially proximal sequences
2. **Data-driven Approach**: Learning human-human interactions from data rather than hand-crafted functions
3. **Joint Prediction**: Simultaneously predicting trajectories of all people in a scene
4. **Superior Performance**: Outperforming traditional social force models and other baselines

## Impact on Social-xLSTM Project

This paper provides the foundational concept for Social Pooling in our Social-xLSTM implementation:

### Core Concepts Adopted
- **Distributed LSTM Architecture**: Each individual (VD) has its own LSTM/xLSTM instance
- **Hidden State Pooling**: Social interactions occur at the hidden state level, not raw features
- **Spatial Awareness**: Neighbor selection based on spatial proximity
- **Grid-based Pooling**: Preserving spatial relationships in the pooling operation

### Adaptations for Traffic Prediction
- **VD-based Architecture**: Traffic detectors (VDs) replace pedestrian trajectories
- **xLSTM Enhancement**: Upgrading from traditional LSTM to xLSTM for better memory capacity
- **Traffic-specific Features**: Adapting input features for traffic flow, speed, and occupancy
- **Scalable Implementation**: Supporting larger numbers of VDs with efficient memory management

### Technical Implementation Mapping

| Social LSTM Concept | Social-xLSTM Implementation |
|-------------------|---------------------------|
| Per-person LSTM | Per-VD xLSTM via VDXLSTMManager |
| Social pooling layer | XLSTMSocialPoolingLayer |
| Grid-based neighbor map | Spatial radius-based neighbor selection |
| Hidden state sharing | Dictionary-based hidden state aggregation |
| Trajectory prediction | Traffic flow prediction |

## References

```bibtex
@inproceedings{alahi2016social,
  title={Social LSTM: Human Trajectory Prediction in Crowded Spaces},
  author={Alahi, Alexandre and Goel, Kratarth and Ramanathan, Vignesh and Robicquet, Alexandre and Fei-Fei, Li and Savarese, Silvio},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={961--971},
  year={2016}
}
```

---

**Note**: This document contains the key concepts and technical details from the original Social LSTM paper that are most relevant to the Social-xLSTM project. For complete mathematical formulations and experimental details, please refer to the original paper.
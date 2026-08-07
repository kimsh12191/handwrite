# Key Papers for ODE-Based Handwriting Synthesis

자세한 분석이 필요한 핵심 논문들

---

## 1. Modeling Trajectories with Neural Ordinary Differential Equations (IJCAI 2021)

**Status:** ⭐⭐⭐ CRITICAL - READ FIRST

### Paper Details
- **Link:** https://www.ijcai.org/proceedings/2021/0207.pdf
- **Focus Area:** Neural ODE application to trajectory modeling
- **Relevance:** 우리 프로젝트의 기초가 될 논문

### Core Concepts
```
Standard RNN approach (your comparison point):
input[t] → RNN cell → hidden_state[t] → output[t]
(Discrete time steps)

Neural ODE approach:
dh/dt = f(h(t), x(t); θ)
h(t) = ODESolve(f, h(0), t; θ)
(Continuous time dynamics)
```

### Key Takeaways for Project
1. **Continuous trajectory representation**: Enables smooth pen trajectory generation
2. **Memory efficiency**: Single continuous function vs. many discrete parameters
3. **Flexibility**: Can evaluate at any time point
4. **Interpretability**: Differential equation parameters have physical meaning

### Implementation Approach
- Use `torchdiffeq` library for ODE solving
- Forward pass: `odeint(f, h0, t_eval)`
- Backward pass: Adjoint method for efficient gradients
- f(h,t) network: 2-3 layer MLP

### Why It Matters for Handwriting
- **Smooth trajectory generation**: Natural pen movement
- **Parameter extraction**: Can extract handwriting dynamics parameters
- **Generalization**: Learn once, generate for any text

---

## 2. Generating Sequences With Recurrent Neural Networks (Graves 2013)

**Status:** ⭐⭐⭐ FOUNDATIONAL - UNDERSTAND THIS FIRST

### Paper Details
- **Link:** https://arxiv.org/pdf/1308.0850
- **Authors:** Alex Graves
- **Key Innovation:** LSTM + Mixture Density Network for trajectory generation

### Architecture Overview
```
Input: text sequence "hello"
    ↓
[Attention + LSTM layers]
    ↓
Output: sequence of (x, y, pen-up/down) points
        with continuous probability distributions
```

### Output Format
For each time step, generate:
```
- x coordinate: continuous (MDN)
- y coordinate: continuous (MDN)
- pen-up/down: discrete (softmax)
- [optional] pressure, angle
```

### Mixture Density Network Explanation
```
Instead of: 
  output = linear(hidden) → single value

Use MDN:
  μ = linear1(hidden) → mean
  σ = linear2(hidden) → std dev
  ρ = linear3(hidden) → mixing coeff
  
  p(y|x) = Σ ρ_i * N(y; μ_i, σ_i)
```

### Key Insights
1. **Attention mechanism**: Focus on which character to write
2. **Mixture models**: Handle multi-modal output distribution (handwriting is stochastic)
3. **Long-term dependencies**: LSTM captures writer's style across sequence

### For Our ODE Project
- Can use MDN output layer with Neural ODE hidden state
- Instead of: h[t] = LSTM(x[t], h[t-1])
- Use: dh/dt = f_ode(h, t; θ), output_params = MDN(h(t))

### Reference Implementation
The paper's code shows clean separation:
- Encoder: Text → attention weights
- Decoder: LSTM → (x,y,pen) coordinates
- Loss: MDN NLL loss

---

## 3. Online Handwriting Trajectory Reconstruction from Kinematic Sensors using TCN (2023)

**Status:** ⭐⭐ PRACTICAL APPLICATION

### Paper Details
- **Link:** https://arxiv.org/abs/2607.26733
- **Focus:** Trajectory reconstruction from IMU sensors
- **Practical Value:** Real-world pen dynamics understanding

### Key Contributions
1. **IMU to trajectory mapping**: Reconstruct pen position from acceleration/gyro
2. **Temporal Convolutional Networks**: Parallel processing alternative to RNNs
3. **Evaluation metrics**: Fréchet distance for trajectory similarity

### Why It Matters
- **Inverse problem**: Goes from dynamics (velocity, acceleration) → coordinates
- **Real sensor data**: Handles noise and missing data
- **Validates trajectory representation**: Ensures our trajectory format is practical

### TCN Architecture Benefits
```
Advantages over RNN:
- Parallel computation
- Constant memory (causal convolutions)
- Larger receptive field possible
- Easier training

For ODE context:
- Can be seen as discretization of continuous dynamics
- Validates that continuous models work for trajectory data
```

---

## 4. Disentangling Writer and Character Styles for Handwriting Generation (2023)

**Status:** ⭐⭐ IMPORTANT FOR PARAMETERIZATION

### Paper Details
- **Link:** https://arxiv.org/pdf/2303.14736
- **Focus:** Style decomposition in handwriting

### Key Concept
```
Handwriting style = Writer style + Character style

Writer style (global parameters θ_writer):
- Slant angle
- Letter spacing
- Overall pressure/speed
- Curvature tendency

Character style (per-char parameters θ_char):
- Height scaling for 'l' vs 'i'
- Baseline positioning
- Serifs/flourishes
- Width variations
```

### For ODE Parameter Design
```
Proposed decomposition:

θ_total = θ_writer (20-25 dims) + θ_char (5-8 dims per character type)

θ_writer includes:
- Oscillation frequency α
- Damping coefficient β  
- Pressure dynamics λ
- Angle tendency κ
- [other dynamics params]

θ_char includes:
- Character-specific amplitude scaling
- Vertical position offset
- Specific shape deformations
```

### Implementation Strategy
1. Learn base ODE parameters from reference handwriting (θ_writer)
2. Fine-tune character offsets (θ_char) for each character
3. Generate new text with θ_writer + appropriate θ_char combinations

---

## 5. DiffInk: Glyph- and Style-Aware Latent Diffusion Transformer (2024)

**Status:** ⭐⭐ STATE-OF-THE-ART GENERATION

### Paper Details
- **Link:** https://arxiv.org/pdf/2509.23624
- **Focus:** Modern diffusion-based handwriting synthesis

### Why Diffusion Models Are Relevant
```
Traditional approach (Graves):
1. Direct regression: Hidden state → coordinates
2. Deterministic + noise during training

Diffusion approach:
1. Learn to denoise trajectories
2. Start from noise → gradually denoise to real trajectory
3. Can sample multiple diverse handwritings

Advantages:
- Better quality generation
- More stable training
- Natural way to handle stochasticity
```

### Architecture Insights
- **Latent space**: Compress trajectories to 16-32D
- **Style encoding**: Condition on reference handwriting
- **Glyph awareness**: Different generation for different characters
- **Transformer decoder**: Generate trajectories autoregressively or in parallel

### Hybrid Approach for Our Project
```
Option 1: ODE-only
- Pro: Interpretable parameters
- Con: May need diffusion for quality

Option 2: ODE + Diffusion
- Phase 1: ODE generates coarse trajectory
- Phase 2: Diffusion refines & adds detail
- Can use ODE parameters as diffusion conditioning

Option 3: ODE as diffusion decoder
- Learn ODE parameters via diffusion
- Use diffusion to learn parameter distribution
```

### Reference for LLM Integration
DiffInk shows how to integrate style conditioning - our LLM can condition ODE on extracted parameters.

---

## 6. Rotation-free Character Recognition Using Linear Recurrent Units (2025)

**Status:** ⭐⭐ STATE-SPACE MODEL ALTERNATIVE

### Paper Details
- **Link:** https://arxiv.org/html/2602.01533
- **Focus:** State-space models (LRU) vs RNNs for sequential data

### Key Comparison
```
RNN (LSTM):
- Nonlinear, expressive
- Hard to train (vanishing gradients)
- Sequential computation

State-Space Model (Mamba, LRU):
- h[t+1] = A @ h[t] + B @ u[t]
- Fast parallel training
- Linear or near-linear in some formulations
- Strong empirical results

Key insight:
State-space models are linear approximations to dynamics!
Can be viewed as discretization of: dh/dt = A @ h + B @ u
```

### Why This Matters
```
Neural ODE: dh/dt = f(h, t)           (nonlinear, continuous)
SSM:        dh/dt = A @ h + B @ u     (linear, continuous)
RNN:        h[t+1] = f(h[t], u[t])    (nonlinear, discrete)

Relationship:
- SSM is simpler version of Neural ODE
- Can start with SSM, progress to nonlinear ODE
- SSM + attention might be sufficient for handwriting
```

### Implementation Strategy
1. **Option A (Recommended for Phase 1):** Use linear state-space model
   - Faster training
   - Interpretable parameters (A matrix encodes dynamics)
   - Can extract handwriting physics easily

2. **Option B (Phase 2):** Upgrade to Neural ODE
   - More expressive
   - Better generation quality
   - Can extract from pretrained SSM

### Concrete State-Space Handwriting Model
```python
# Linear state-space formulation:
# h[t+1] = A @ h[t] + B @ (text_feature[t])
# y[t] = C @ h[t] + D @ (text_feature[t])

# Where:
# - h[t]: handwriting state (20-30 dims)
# - A: learned dynamics matrix (20×20)
# - B: text input matrix (20×768)
# - C,D: output projection
# - y[t]: (x, y, pressure) coordinates
```

---

## 7. Comprehensive Review of Neural Differential Equations (2025)

**Status:** ⭐⭐ THEORETICAL FOUNDATION

### Paper Details
- **Link:** https://arxiv.org/html/2502.09885v1
- **Focus:** Survey of Neural ODE variants and theory

### Important Variants for Handwriting

#### 1. Standard Neural ODE
```
dh/dt = f_θ(h, t)
y(t) = g(h(t))
```
- Most general
- Most expressive
- Slowest (ODE solver needed)

#### 2. Neural Controlled ODE
```
dh/dt = f_θ(h, u(t), t)
```
- Condition on external signals (text embeddings)
- Better for sequential control
- **MOST RELEVANT FOR HANDWRITING**

#### 3. Gated Neural ODE
```
dh/dt = f_θ(h, t) ⊙ g_θ(h, t)
```
- Learned gates for dynamic activation
- Prevents explosion/vanishing
- More stable training

#### 4. Latent ODE
```
z(t) = ODESolve(f, z_0, t)
x(t) = decoder(z(t))
```
- Compress observation to latent ODE
- Useful with high-dimensional output (images)
- Can combine with VAE

### For Handwriting Project
```
Recommended architecture:

Input: text embedding + image (for inverse problem)
    ↓
Encoder: feature_vector = encode(text, image)
    ↓
Neural Controlled ODE:
  dh/dt = f_θ(h, feature_vector, t)
  h(0) = initialize(feature_vector)
    ↓
Output layer (MDN):
  μ_x, σ_x, μ_y, σ_y, ... = MLP(h(t))
    ↓
Trajectory: (x, y, pen, ...) coordinates
```

---

## Implementation Roadmap

### Phase 1: Linear State-Space Model (Week 1-2)
```
Simplest approach that works:
- h[t+1] = A @ h[t] + B @ features[t]
- Extract A matrix as "handwriting dynamics"
- Fast training, interpretable parameters
```

### Phase 2: Neural ODE (Week 3-4)
```
Upgrade to full Neural ODE:
- dh/dt = f_θ(h, features, t)
- Better quality trajectories
- Can initialize from Phase 1
```

### Phase 3: Generative Model + LLM Loop (Week 5-6)
```
Add LLM inference:
- LLM → parameter prediction
- Simulator → trajectory
- VLM → comparison
- Auto-refinement loop
```

---

## Critical Equations to Implement

### 1. ODE Dynamics
```
Position: ẍ = -α*ẋ - β*x + γ*sin(ωt + φ) + c_control
Pressure: ṗ = -λ*p + μ + σ*v
Angle:    θ̇ = κ*arctan(ẏ/ẋ) + ρ
```

### 2. Mixture Density Network Loss
```
MDN_loss = -log Σ_k π_k * N(y; μ_k, σ_k)
where π_k: mixing probability
      μ_k, σ_k: mean, std dev of component k
```

### 3. ODE Solution via torchdiffeq
```python
h_traj = odeint(f, h_0, t_eval)
```

### 4. Trajectory Rendering
```
SVG path generation or rasterization
to pixel image
```

---

## Recommended Reference Code

1. **Graves implementation:** 
   - Search GitHub for "handwriting-synthesis-graves"
   - Shows LSTM + MDN architecture

2. **Neural ODE reference:**
   - `torchdiffeq` library examples
   - ODE-VAE for inspiration

3. **State-space models:**
   - Mamba: https://github.com/state-spaces/mamba
   - S4: https://github.com/state-spaces/s4

---

## Next Steps

1. **Read Priority 1 papers** (2-3 days)
   - Fully understand Neural ODE concepts
   - Understand MDN trajectory generation
   - Review current SOTA

2. **Design ODE model** (1 day)
   - Choose state dimension (20-30)
   - Define f_θ architecture
   - Plan parameter interpretation

3. **Prototype Phase 1** (1 week)
   - Linear state-space model
   - Simple trajectory generation
   - Verify rendering pipeline

4. **Expand to Neural ODE** (1 week)
   - Switch to neural dynamics
   - Improve generation quality
   - Add parameter extraction

5. **LLM integration** (1-2 weeks)
   - Inverse problem: image → parameters
   - Auto-research loop
   - VLM feedback integration

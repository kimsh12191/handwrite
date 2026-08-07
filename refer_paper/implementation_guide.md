# Implementation Guide: ODE-Based Handwriting Synthesis

Phase 1-3를 위한 구현 전략 및 기술 선택

---

## Technology Stack

### Core Libraries
```
Python 3.10+
PyTorch 2.0+
scipy >= 1.10.0
matplotlib (visualization)
Pillow (image rendering)

Optional for LLM:
- transformers (for VLM if needed)
- requests (for 397B model API)
```

### ODE Solving
```
Primary: torchdiffeq
- pip install torchdiffeq
- Supports various ODE solvers (RK45, dopri5, etc.)
- Backprop through ODE solutions

Fallback: scipy.integrate.odeint
- For prototyping without GPU
```

---

## Architecture Decision Tree

### Step 1: Choose State Representation
```
Question: How much detail in handwriting dynamics?

Option A: Minimal (2D position only)
  State: h = [x, y]
  ODE: ẋ=vx, ẏ=vy, v̇x=f_a(...), v̇y=f_a(...)
  Complexity: Simple
  Expressiveness: Low-Medium
  USE WHEN: Fast prototyping

Option B: Recommended (position + pressure + angle)
  State: h = [x, y, vx, vy, p, θ] (6D)
  Complexity: Medium
  Expressiveness: Medium-High
  USE WHEN: Phase 1-2 (Recommended)

Option C: Full (position + velocity + pressure + angle + derivatives)
  State: h = [x, y, vx, vy, ax, ay, p, θ] (8D)
  Complexity: Higher
  Expressiveness: High
  USE WHEN: Phase 3 (refinement)
```

**Recommendation for Phase 1: Option B (6D state)**

### Step 2: Choose Dynamics Model
```
Option 1: Learned f_θ(h, t; θ) - Pure Neural ODE

  dh/dt = MLP(h, t)
  
  Pros:
  - Most flexible
  - Learns whatever dynamics exist
  
  Cons:
  - Hard to interpret parameters
  - May need diffusion for quality
  
  USE WHEN: Full generality needed

Option 2: Parameterized physics-inspired (Recommended)

  ẋ = vx
  ẏ = vy
  v̇x = -α*vx - β*x + γ*sin(ωt) + c_text[0]
  v̇y = -α*vy - β*y + γ*sin(ωt) + c_text[1]
  ṗ = -λ*p + μ + σ*v
  θ̇ = κ*arctan(vy/vx) + ρ
  
  Pros:
  - Interpretable parameters
  - Physics-grounded
  - Fewer parameters needed
  - Can extract from real handwriting
  
  Cons:
  - Less flexible
  - Manual equation design
  
  USE WHEN: Want interpretability + parameter extraction

Option 3: Hybrid (Physics + learned terms)

  ẋ = vx
  ẏ = vy
  v̇x = -α*vx - β*x + f_learned(h, t; θ)
  
  Pros:
  - Balance between flexibility and interpretability
  - Can start with physics, add learned terms
  
  USE WHEN: Phase 2-3 after physics model works

Option 4: State-Space Model (Linear)

  h[t+1] = A @ h[t] + B @ u[t]
  
  Where u[t] = text_embedding[t] or control signal
  
  Pros:
  - Fastest to train
  - Linear = discretization of continuous linear ODE
  - Interpretable A matrix
  
  Cons:
  - Linear only
  - May need multiple layers for expressiveness
  
  USE WHEN: Phase 1 quickstart
```

**Recommendation for Phase 1: Option 2 (Parameterized physics-inspired)**
- Clear interpretation of parameters
- Can extract from real handwriting images
- Enables LLM reasoning about parameters

---

## Phase 1: Linear State-Space / Simple ODE Model

### Architecture
```python
class HandwritingDynamics:
    def __init__(self, state_dim=6):
        # State: [x, y, vx, vy, p, θ]
        self.state_dim = state_dim
        
    def f_dynamics(self, state, t, text_features):
        """
        dh/dt = f(h, t, text_features; θ)
        
        Returns time derivatives of state
        """
        x, y, vx, vy, p, theta = state
        
        # Position derivatives (velocity)
        dx_dt = vx
        dy_dt = vy
        
        # Velocity dynamics (damped oscillator + text control)
        dvx_dt = -self.alpha*vx - self.beta*x + \
                 self.gamma*sin(self.omega*t) + \
                 self.c_text[0]*text_features
        dvy_dt = -self.alpha*vy - self.beta*y + \
                 self.gamma*sin(self.omega*t) + \
                 self.c_text[1]*text_features
        
        # Pressure dynamics
        dp_dt = -self.lambda_p*p + self.mu_p + self.sigma_p*sqrt(vx**2 + vy**2)
        
        # Angle dynamics
        dtheta_dt = self.kappa*arctan(vy/vx) + self.rho
        
        return [dx_dt, dy_dt, dvx_dt, dvy_dt, dp_dt, dtheta_dt]

# Learnable parameters (20-25 total):
# α, β, γ, ω, φ (oscillation)        : 5
# λ, μ, σ (pressure)                : 3
# κ, ρ (angle)                       : 2
# c_text vectors                     : 2 (control from text embedding)
# + Character-specific modifiers     : 5-8
```

### Training Loop
```python
import torch
from torchdiffeq import odeint

class HandwritingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dynamics = HandwritingDynamics()
        self.encoder = TextEncoder()  # text → features
        
    def forward(self, text, t_eval):
        # Encode text
        text_features = self.encoder(text)
        
        # Initial state
        h0 = self.init_state(text_features)
        
        # Solve ODE
        h_traj = odeint(
            func=lambda t, h: self.dynamics.f(h, t, text_features),
            y0=h0,
            t=t_eval,
            method='dopri5'  # or other solver
        )
        
        return h_traj
    
    def trajectory_to_image(self, h_traj):
        # Extract x, y, pressure from state
        # Render to image
        pass

# Loss function
def loss_fn(generated_traj, target_traj):
    # Trajectory loss (smoothness + accuracy)
    traj_loss = mse(generated_traj[:, :2], target_traj[:, :2])
    
    # Optional: smoothness penalty
    smooth_loss = mse(generated_traj[1:] - generated_traj[:-1], 0)
    
    # Optional: dynamics regularization
    dynamics_loss = regularize_parameters(model.dynamics)
    
    return traj_loss + λ*smooth_loss + μ*dynamics_loss
```

---

## Phase 2: Full Neural ODE

### When to Switch
```
Transition checklist:
✓ Phase 1 model generates reasonable trajectories
✓ Parameter extraction working from real handwriting
✓ Loss convergents smoothly
✓ Basic simulator/renderer working
✓ Can generate 3-5 different styles with different parameters
```

### Architecture Upgrade
```python
class NeuralODEDynamics(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Stack of learned functions
        self.f1 = nn.Sequential(
            nn.Linear(6 + 8, 64),  # 6D state + 8D text features
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 6)       # 6D time derivatives
        )
        # Optional: gated activation
        self.gate = nn.Linear(64, 6)
    
    def forward(self, t, h, text_features):
        # Concatenate state and control
        h_aug = torch.cat([h, text_features.repeat(h.shape[0], 1)], dim=-1)
        
        # Learned dynamics
        h_dot = self.f1(h_aug)
        
        # Optional gating for stability
        # h_dot = h_dot * sigmoid(self.gate(...))
        
        return h_dot
```

### Hybrid Approach: Start with Physics, Add Learned Terms
```python
class HybridODEDynamics:
    """
    Combine physics-based structure with learned refinements
    """
    def __init__(self):
        self.physics = HandwritingDynamics()
        self.learner = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )
    
    def forward(self, t, h, text_features):
        # Physics-based part
        h_dot_physics = self.physics.f(h, t, text_features)
        
        # Learned correction
        h_dot_learned = self.learner(h)
        
        # Combine
        return h_dot_physics + λ*h_dot_learned
```

---

## Phase 3: LLM Integration for Inverse Inference

### Architecture
```
Image input
    ↓
[VLM encoder] → visual features
    ↓
[Text context] → text embedding
    ↓
[LLM] "Given these visual features and this text,
       what are the ODE parameters?"
    ↓
[Parameter vector output] → θ = [α, β, γ, ...]
    ↓
[Simulator] Generate trajectory with θ
    ↓
[VLM comparison] Compare generated vs original
    ↓
[LLM] Refine parameter estimate
    ↓
[Repeat until convergence]
```

### Prompt Structure for LLM
```
System Prompt:
"You are an expert in handwriting dynamics and ordinary differential equations.
Given a handwritten text image and the text content,
analyze the handwriting style and extract its dynamic parameters.

The handwriting follows ODE:
ẋ = vx
ẏ = vy
v̇x = -α*vx - β*x + γ*sin(ωt) + c_text
...

Parameters to estimate: α (damping), β (restoring force), γ (oscillation amplitude), ..."

User Message:
"Image: [visual features from VLM]
Text: 'hello'
Current estimate: α=0.2, β=0.1, γ=0.05, ...
VLM feedback: Generated trajectory has too much oscillation, angle is wrong

Refine the parameter estimate. Return updated values in JSON format:
{
  "alpha": <number>,
  "beta": <number>,
  ...
}"
```

### Refinement Loop
```python
class AutoResearchLoop:
    def __init__(self, llm, vlm, simulator):
        self.llm = llm
        self.vlm = vlm
        self.simulator = simulator
        self.max_iterations = 5
    
    def refine_parameters(self, target_image, text):
        # Initial parameter estimation
        visual_features = self.vlm.encode(target_image)
        
        params = self.llm.estimate_parameters(
            visual_features=visual_features,
            text=text
        )
        
        for iteration in range(self.max_iterations):
            # Generate trajectory
            generated_traj = self.simulator.generate(params, text)
            generated_image = self.simulator.render(generated_traj)
            
            # Get VLM feedback
            feedback = self.vlm.compare(
                target=target_image,
                generated=generated_image
            )
            
            # Refine parameters based on feedback
            params = self.llm.refine_parameters(
                current_params=params,
                feedback=feedback,
                text=text
            )
            
            # Check convergence
            if self.has_converged(params, feedback):
                break
        
        return params

def has_converged(self, params, feedback):
    # Criteria: feedback indicates "looks good" or parameter changes < threshold
    return "looks good" in feedback or max(param_changes) < 0.01
```

---

## Key Implementation Decisions

### ODE Solver Choice
```
Solver      | Speed | Accuracy | Memory | Best For
dopri5      | Fast  | Good     | Medium | Default choice
adams       | Fast  | Good     | Low    | Long trajectories
rk4         | Medium| Exact    | Medium | Testing/validation
vcabm       | Very  | Good     | Low    | Large batches
           | Fast

Recommendation: Start with dopri5, test with adams for long sequences
```

### Batch Processing
```
Option 1: Batch of trajectories with same ODE parameters
- Generate multiple texts with one writer's parameters
- Efficient: Solve ODE once, evaluate at multiple text points

Option 2: Batch of different parameters
- Different writer styles in parallel
- Use vmap() or batch matrix operations

Option 3: Mixed batch
- Different texts + different parameters
- Most flexible but slowest
```

### Loss Function Components
```
Total Loss = α*L_trajectory + β*L_smoothness + γ*L_dynamics + δ*L_regularization

L_trajectory: 
  MSE between generated and target coordinates
  
L_smoothness:
  MSE(trajectory[t+1] - trajectory[t]) - penalize jerky motion
  
L_dynamics:
  How well does the ODE fit the trajectory?
  Only used during parameter extraction phase
  
L_regularization:
  Penalties on parameter magnitudes
  Smoothness of parameter variations
```

---

## Rendering Pipeline

### Trajectory to Image
```python
def render_trajectory(trajectory, width=400, height=300):
    """
    trajectory: np.array shape (T, 6) with [x, y, vx, vy, p, θ]
    returns: PIL Image
    """
    img = Image.new('RGB', (width, height), color='white')
    
    # Normalize coordinates to image space
    x_norm = (trajectory[:, 0] - min_x) / (max_x - min_x) * width
    y_norm = (trajectory[:, 1] - min_y) / (max_y - min_y) * height
    pressure = trajectory[:, 4]  # 0-1
    
    # Draw strokes
    draw = ImageDraw.Draw(img)
    for i in range(len(trajectory)-1):
        if pressure[i] > 0.1:  # Pen is down
            width_px = int(1 + pressure[i] * 4)  # 1-5 pixels
            draw.line(
                [(x_norm[i], y_norm[i]), (x_norm[i+1], y_norm[i+1])],
                fill='black',
                width=width_px
            )
    
    return img
```

### SVG Alternative (Scalable)
```python
def render_trajectory_svg(trajectory, filename, width=400, height=300):
    """
    SVG rendering for scalability
    """
    import svgwrite
    
    dwg = svgwrite.Drawing(filename, size=(width, height))
    
    x_norm = (trajectory[:, 0] - min_x) / (max_x - min_x) * width
    y_norm = (trajectory[:, 1] - min_y) / (max_y - min_y) * height
    pressure = trajectory[:, 4]
    
    for i in range(len(trajectory)-1):
        if pressure[i] > 0.1:
            width_px = 1 + pressure[i] * 4
            line = dwg.line(
                start=(x_norm[i], y_norm[i]),
                end=(x_norm[i+1], y_norm[i+1]),
                stroke='black',
                stroke_width=width_px
            )
            dwg.add(line)
    
    dwg.save()
```

---

## Testing & Validation

### Unit Tests
```python
def test_ode_solution_shapes():
    model = HandwritingModel()
    text = "hello"
    t_eval = np.linspace(0, 1, 100)
    
    h_traj = model(text, t_eval)
    assert h_traj.shape == (100, 6)  # T × state_dim

def test_trajectory_continuity():
    # Verify ODE solution is smooth
    h_traj = model(text, t_eval)
    
    # Acceleration should be smooth
    vel = np.diff(h_traj[:, :2], axis=0)
    acc = np.diff(vel, axis=0)
    
    # Second derivative smoothness (curvature)
    assert np.max(np.abs(np.diff(acc, axis=0))) < threshold

def test_rendering():
    h_traj = model(text, t_eval)
    img = render_trajectory(h_traj)
    
    # Check output is valid image
    assert img.size == (400, 300)
    assert np.min(np.array(img)) >= 0
```

### Integration Tests
```python
def test_parameter_extraction_roundtrip():
    """
    Original trajectory → Extract parameters → 
    Generate with params → Should match original
    """
    original_params = {"alpha": 0.3, "beta": 0.1, ...}
    
    # Generate trajectory
    original_traj = simulator.generate(original_params, "hello")
    original_img = render(original_traj)
    
    # Extract parameters from image
    extracted_params = extract_parameters(original_img, "hello")
    
    # Regenerate
    regenerated_traj = simulator.generate(extracted_params, "hello")
    regenerated_img = render(regenerated_traj)
    
    # Should be similar
    ssim = compute_ssim(original_img, regenerated_img)
    assert ssim > 0.9
```

---

## Performance Optimization

### GPU Acceleration
```python
# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Batch processing on GPU
h_traj = odeint(..., method='dopri5')  # Runs on GPU by default

# Forward/backward pass automatically uses GPU
```

### Memory Management
```python
# For long sequences, use checkpointing
from torch.utils.checkpoint import checkpoint

def solve_with_checkpoint(model, y0, t):
    return checkpoint(
        lambda y: odeint(model, y, t),
        y0,
        use_reentrant=False
    )
```

### Parallelization
```python
# Generate multiple texts with same parameters (vmap)
import torch.func as F

def generate_batch(params, texts):
    return F.vmap(
        lambda text: simulator.generate(params, text)
    )(texts)
```

---

## Debugging Guide

### Common Issues

1. **ODE solution diverges or NaNs**
   - Reduce step size (solver tolerance)
   - Add regularization to parameters
   - Check dynamics function for instabilities
   
2. **Trajectories are too noisy**
   - Add smoothness penalty to loss
   - Increase damping coefficient α
   - Use smoother solver (e.g., adams)

3. **Parameter extraction fails**
   - Trajectory might not be invertible
   - Try fewer parameters first
   - Add more constraints

4. **LLM parameter predictions are wrong**
   - Provide better visual features to LLM
   - Give more examples in prompt
   - Add validation step: generate and compare

---

## Next Steps

1. **Implement Phase 1** this week
   - Set up ODE solver
   - Define physics-based dynamics
   - Create simple simulator
   - Build renderer

2. **Validate Phase 1** (1 week)
   - Test different parameter combinations
   - Verify trajectory smoothness
   - Compare with existing methods if possible

3. **Start Phase 2** (Week 3)
   - Implement Neural ODE upgrade
   - Compare quality with Phase 1
   - Keep interpretability for LLM integration

4. **Plan Phase 3 integration** (Week 4)
   - Design LLM prompts
   - Set up VLM encoder
   - Build auto-research loop


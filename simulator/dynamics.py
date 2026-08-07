"""
Physics-based handwriting dynamics model.

State vector: h = [x, y, vx, vy, p, theta]
  x, y: pen position
  vx, vy: pen velocity
  p: pressure (0-1)
  theta: pen angle
"""

import numpy as np
import torch
import torch.nn as nn


class HandwritingDynamicsParams:
    """Container for handwriting dynamics parameters."""

    def __init__(self):
        # Velocity dynamics (damped oscillator)
        self.alpha = 0.3      # velocity damping
        self.beta = 0.1       # restoring force (position)
        self.gamma = 0.05     # oscillation amplitude
        self.omega = 2.0      # oscillation frequency

        # Pressure dynamics
        self.lambda_p = 0.5   # pressure decay rate
        self.mu_p = 0.3       # baseline pressure
        self.sigma_p = 0.1    # pressure modulation by velocity

        # Angle dynamics
        self.kappa = 0.5      # angle tracking of velocity
        self.rho = 0.0        # angle offset

        # Text control
        self.c_text_x = 0.1   # x control from text features
        self.c_text_y = 0.1   # y control from text features

    @staticmethod
    def from_dict(d):
        """Create from dictionary."""
        params = HandwritingDynamicsParams()
        for key, val in d.items():
            if hasattr(params, key):
                setattr(params, key, val)
        return params

    def to_dict(self):
        """Convert to dictionary."""
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'omega': self.omega,
            'lambda_p': self.lambda_p,
            'mu_p': self.mu_p,
            'sigma_p': self.sigma_p,
            'kappa': self.kappa,
            'rho': self.rho,
            'c_text_x': self.c_text_x,
            'c_text_y': self.c_text_y,
        }


class HandwritingDynamics(nn.Module):
    """
    ODE dynamics for handwriting synthesis.

    dh/dt = f(h, t, text_features; params)

    where h = [x, y, vx, vy, p, theta]
    """

    def __init__(self, params=None):
        super().__init__()

        if params is None:
            params = HandwritingDynamicsParams()

        # Register parameters as learnable tensors
        self.register_parameter('alpha', nn.Parameter(torch.tensor(params.alpha)))
        self.register_parameter('beta', nn.Parameter(torch.tensor(params.beta)))
        self.register_parameter('gamma', nn.Parameter(torch.tensor(params.gamma)))
        self.register_parameter('omega', nn.Parameter(torch.tensor(params.omega)))

        self.register_parameter('lambda_p', nn.Parameter(torch.tensor(params.lambda_p)))
        self.register_parameter('mu_p', nn.Parameter(torch.tensor(params.mu_p)))
        self.register_parameter('sigma_p', nn.Parameter(torch.tensor(params.sigma_p)))

        self.register_parameter('kappa', nn.Parameter(torch.tensor(params.kappa)))
        self.register_parameter('rho', nn.Parameter(torch.tensor(params.rho)))

        self.register_parameter('c_text_x', nn.Parameter(torch.tensor(params.c_text_x)))
        self.register_parameter('c_text_y', nn.Parameter(torch.tensor(params.c_text_y)))

    def forward(self, t, h, text_features=None):
        """
        Compute time derivatives of state.

        Args:
            t: scalar time (for oscillation)
            h: state vector [..., 6] with [..., 0:2] = position, [2:4] = velocity, [4] = pressure, [5] = angle
            text_features: optional control signal [..., n_features]

        Returns:
            h_dot: time derivatives with same shape as h
        """
        # Unpack state
        x = h[..., 0:1]
        y = h[..., 1:2]
        vx = h[..., 2:3]
        vy = h[..., 3:4]
        p = h[..., 4:5]
        theta = h[..., 5:6]

        # Position derivatives (velocity)
        dx_dt = vx
        dy_dt = vy

        # Oscillation term
        oscillation = self.gamma * torch.sin(self.omega * t)

        # Text control (optional)
        text_control_x = 0.0
        text_control_y = 0.0
        if text_features is not None:
            # Simple: use first dimension of text features
            if text_features.dim() == 0:
                text_feat = text_features.unsqueeze(0)
            else:
                text_feat = text_features[..., :1]
            text_control_x = self.c_text_x * text_feat
            text_control_y = self.c_text_y * text_feat

        # Velocity dynamics (damped nonlinear oscillator)
        # v̇ = -α*v - β*x + γ*sin(ωt) + text_control
        dvx_dt = -self.alpha * vx - self.beta * x + oscillation + text_control_x
        dvy_dt = -self.alpha * vy - self.beta * y + oscillation + text_control_y

        # Pressure dynamics
        # ṗ = -λ*p + μ + σ*√(vx² + vy²)
        speed = torch.sqrt(vx**2 + vy**2 + 1e-8)
        dp_dt = -self.lambda_p * p + self.mu_p + self.sigma_p * speed

        # Angle dynamics
        # θ̇ = κ*atan2(vy, vx) + ρ
        angle_target = torch.atan2(vy, vx + 1e-8)
        dtheta_dt = self.kappa * (angle_target - theta) + self.rho

        # Stack derivatives
        h_dot = torch.cat([dx_dt, dy_dt, dvx_dt, dvy_dt, dp_dt, dtheta_dt], dim=-1)

        return h_dot

    def get_params_dict(self):
        """Get current parameters as dictionary."""
        return {
            'alpha': self.alpha.item(),
            'beta': self.beta.item(),
            'gamma': self.gamma.item(),
            'omega': self.omega.item(),
            'lambda_p': self.lambda_p.item(),
            'mu_p': self.mu_p.item(),
            'sigma_p': self.sigma_p.item(),
            'kappa': self.kappa.item(),
            'rho': self.rho.item(),
            'c_text_x': self.c_text_x.item(),
            'c_text_y': self.c_text_y.item(),
        }


def create_default_dynamics():
    """Create dynamics with default parameters."""
    params = HandwritingDynamicsParams()
    params.alpha = 0.3
    params.beta = 0.1
    params.gamma = 0.05
    params.omega = 2.0
    params.lambda_p = 0.5
    params.mu_p = 0.3
    params.sigma_p = 0.1

    return HandwritingDynamics(params)

"""
Handwriting synthesis model using neural ODE solver.
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np

from .dynamics import HandwritingDynamics, HandwritingDynamicsParams


class TextEncoder(nn.Module):
    """Simple text encoder for character embeddings."""

    def __init__(self, vocab_size=256, embedding_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, text):
        """
        Args:
            text: string or list of character indices

        Returns:
            embeddings: [seq_len, embedding_dim]
        """
        if isinstance(text, str):
            # Convert string to indices
            indices = torch.tensor([ord(c) % 256 for c in text], dtype=torch.long)
        else:
            indices = text

        return self.embedding(indices)


class HandwritingModel(nn.Module):
    """
    End-to-end handwriting synthesis model.

    Solves ODE: dh/dt = f(h, t, text_features; θ)
    """

    def __init__(self, embedding_dim=16, device='cpu'):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim

        # Text encoder
        self.text_encoder = TextEncoder(embedding_dim=embedding_dim)

        # ODE dynamics
        self.dynamics = HandwritingDynamics()

        # Initial state network
        self.init_state_net = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 6)  # 6D state
        )

        # Move to device
        self.to(device)

    def forward(self, text, t_eval, method='dopri5', return_full_traj=True):
        """
        Generate handwriting trajectory for given text.

        Args:
            text: string to write
            t_eval: time points to evaluate at [T]
            method: ODE solver method ('dopri5', 'adams', 'rk4')
            return_full_traj: if True, return full trajectory; else just final state

        Returns:
            trajectory: [T, 6] with state at each time point
        """
        # Encode text
        text_features = self.text_encoder(text)  # [seq_len, embedding_dim]

        # Compute initial state from mean of text features
        text_mean = text_features.mean(dim=0, keepdim=True)  # [1, embedding_dim]
        h0 = self.init_state_net(text_mean)  # [1, 6]
        h0 = h0.squeeze(0)  # [6]

        # Ensure t_eval is on correct device
        if isinstance(t_eval, np.ndarray):
            t_eval = torch.from_numpy(t_eval).float().to(self.device)
        else:
            t_eval = t_eval.float().to(self.device)

        # ODE solver
        with torch.no_grad():
            trajectory = odeint(
                func=lambda t, h: self.dynamics(t, h, text_features),
                y0=h0,
                t=t_eval,
                method=method,
                rtol=1e-3,
                atol=1e-4,
            )

        return trajectory

    def generate_trajectory(self, text, num_points=200, duration=2.0):
        """
        Convenience method to generate trajectory.

        Args:
            text: string to write
            num_points: number of time steps
            duration: total duration in arbitrary time units

        Returns:
            trajectory: [num_points, 6] numpy array
        """
        t_eval = np.linspace(0, duration, num_points)
        trajectory = self.forward(text, t_eval)
        return trajectory.detach().cpu().numpy()

    def set_dynamics_params(self, params_dict):
        """Update dynamics parameters."""
        for key, val in params_dict.items():
            if hasattr(self.dynamics, key):
                with torch.no_grad():
                    param = getattr(self.dynamics, key)
                    param.copy_(torch.tensor(val))

    def get_dynamics_params(self):
        """Get current dynamics parameters."""
        return self.dynamics.get_params_dict()


class SimpleHandwritingSimulator:
    """
    Wrapper for easy handwriting generation.
    """

    def __init__(self, device='cpu'):
        self.device = device
        self.model = HandwritingModel(device=device)

    def generate(self, text, num_points=200, duration=2.0, style_params=None):
        """
        Generate handwriting for given text.

        Args:
            text: string to write
            num_points: number of time steps
            duration: total duration
            style_params: optional dict of dynamics parameters to override

        Returns:
            trajectory: [num_points, 6] numpy array
        """
        if style_params is not None:
            self.model.set_dynamics_params(style_params)

        return self.model.generate_trajectory(text, num_points, duration)

    def get_styles(self):
        """Get dictionary of predefined styles."""
        return {
            'normal': {
                'alpha': 0.3,
                'beta': 0.1,
                'gamma': 0.05,
                'mu_p': 0.3,
            },
            'shaky': {
                'alpha': 0.2,
                'beta': 0.08,
                'gamma': 0.1,
                'mu_p': 0.4,
            },
            'flowing': {
                'alpha': 0.4,
                'beta': 0.05,
                'gamma': 0.03,
                'mu_p': 0.2,
            },
            'heavy': {
                'alpha': 0.25,
                'beta': 0.12,
                'gamma': 0.04,
                'mu_p': 0.6,
            },
        }

    def generate_with_style(self, text, style_name='normal', num_points=200, duration=2.0):
        """Generate with predefined style."""
        styles = self.get_styles()
        if style_name not in styles:
            raise ValueError(f"Unknown style: {style_name}")

        return self.generate(text, num_points, duration, styles[style_name])

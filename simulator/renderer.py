"""
Trajectory rendering to images.
"""

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


class HandwritingRenderer:
    """Render handwriting trajectories to images."""

    def __init__(self, width=400, height=300, bg_color='white', pen_color='black'):
        self.width = width
        self.height = height
        self.bg_color = bg_color
        self.pen_color = pen_color

    def render_pil(self, trajectory, normalize=True):
        """
        Render trajectory to PIL Image.

        Args:
            trajectory: [T, 6] with state [x, y, vx, vy, p, theta]
            normalize: if True, normalize coordinates to image space

        Returns:
            PIL Image object
        """
        # Extract position and pressure
        x = trajectory[:, 0]
        y = trajectory[:, 1]
        pressure = trajectory[:, 4]  # 0-1

        # Normalize coordinates
        if normalize:
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()

            # Add margin
            x_range = x_max - x_min + 1e-8
            y_range = y_max - y_min + 1e-8
            margin = 0.1

            x_norm = ((x - x_min) / x_range) * (1 - 2 * margin) + margin
            y_norm = ((y - y_min) / y_range) * (1 - 2 * margin) + margin

            x_pixels = x_norm * self.width
            y_pixels = y_norm * self.height
        else:
            x_pixels = x
            y_pixels = y

        # Create image
        img = Image.new('RGB', (self.width, self.height), color=self.bg_color)
        draw = ImageDraw.Draw(img)

        # Draw strokes
        for i in range(len(trajectory) - 1):
            p = float(pressure[i])

            # Only draw if pressure is above threshold
            if p > 0.05:
                x0, y0 = int(x_pixels[i]), int(y_pixels[i])
                x1, y1 = int(x_pixels[i + 1]), int(y_pixels[i + 1])

                # Line width based on pressure
                line_width = max(1, int(1 + p * 5))

                # Draw line
                draw.line([(x0, y0), (x1, y1)], fill=self.pen_color, width=line_width)

        return img

    def render_numpy(self, trajectory, normalize=True):
        """
        Render trajectory to numpy array.

        Args:
            trajectory: [T, 6] array
            normalize: if True, normalize coordinates

        Returns:
            numpy array [height, width, 3] with values 0-255
        """
        img = self.render_pil(trajectory, normalize=normalize)
        return np.array(img)

    def render_matplotlib(self, trajectory, normalize=True, figsize=(6, 4.5)):
        """
        Render trajectory using matplotlib (shows velocity/pressure info).

        Args:
            trajectory: [T, 6] array
            normalize: if True, normalize coordinates
            figsize: figure size

        Returns:
            matplotlib figure
        """
        x = trajectory[:, 0]
        y = trajectory[:, 1]
        vx = trajectory[:, 2]
        vy = trajectory[:, 3]
        p = trajectory[:, 4]

        if normalize:
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            x = (x - x_min) / (x_max - x_min + 1e-8)
            y = (y - y_min) / (y_max - y_min + 1e-8)

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Trajectory plot
        ax = axes[0]
        scatter = ax.scatter(x, y, c=p, cmap='viridis', s=20, alpha=0.6)
        ax.plot(x, y, 'k-', alpha=0.2, linewidth=0.5)
        ax.set_aspect('equal')
        ax.set_title('Trajectory')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(scatter, ax=ax, label='Pressure')

        # Velocity plot
        ax = axes[1]
        speed = np.sqrt(vx**2 + vy**2)
        ax.plot(speed, 'b-', linewidth=1)
        ax.set_title('Velocity (Speed)')
        ax.set_xlabel('Time')
        ax.set_ylabel('Speed')
        ax.grid(True, alpha=0.3)

        # Pressure plot
        ax = axes[2]
        ax.plot(p, 'r-', linewidth=1)
        ax.set_title('Pressure')
        ax.set_xlabel('Time')
        ax.set_ylabel('Pressure')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def trajectories_to_grid(trajectories, grid_shape=(2, 3), figsize=(10, 6)):
        """
        Render multiple trajectories in a grid.

        Args:
            trajectories: list of trajectory arrays or (trajectory, text) tuples
            grid_shape: (rows, cols)
            figsize: figure size

        Returns:
            matplotlib figure
        """
        rows, cols = grid_shape
        fig, axes = plt.subplots(rows, cols, figsize=figsize)

        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.flatten()

        renderer = HandwritingRenderer(width=200, height=150)

        for idx, ax in enumerate(axes):
            if idx < len(trajectories):
                if isinstance(trajectories[idx], tuple):
                    traj, label = trajectories[idx]
                else:
                    traj = trajectories[idx]
                    label = f"Sample {idx}"

                img = renderer.render_numpy(traj, normalize=True)
                ax.imshow(img)
                ax.set_title(label)
            ax.axis('off')

        plt.tight_layout()
        return fig


def render_comparison(target_traj, generated_traj, target_img=None, generated_img=None):
    """
    Render side-by-side comparison of target and generated handwriting.

    Args:
        target_traj: target trajectory [T, 6]
        generated_traj: generated trajectory [T, 6]
        target_img: optional PIL Image of target
        generated_img: optional PIL Image of generated

    Returns:
        matplotlib figure
    """
    renderer = HandwritingRenderer(width=300, height=250)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Target trajectory
    if target_img is None:
        target_img = renderer.render_numpy(target_traj, normalize=True)
    axes[0].imshow(target_img)
    axes[0].set_title('Target')
    axes[0].axis('off')

    # Generated trajectory
    if generated_img is None:
        generated_img = renderer.render_numpy(generated_traj, normalize=True)
    axes[1].imshow(generated_img)
    axes[1].set_title('Generated')
    axes[1].axis('off')

    # Difference (simple overlay)
    overlay = np.ones_like(target_img)
    overlay[generated_img == 255] = [255, 200, 200]  # Generated only - red
    overlay[target_img == 255] = [200, 200, 255]     # Target only - blue
    both = (target_img != 255) & (generated_img != 255)
    overlay[both] = [100, 100, 100]                  # Both - gray

    axes[2].imshow(overlay)
    axes[2].set_title('Overlay (Blue: Target, Red: Generated)')
    axes[2].axis('off')

    plt.tight_layout()
    return fig

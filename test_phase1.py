#!/usr/bin/env python3
"""
Phase 1 Testing Script: Basic handwriting generation with physics-based ODE.

Tests:
1. ODE solver execution
2. Trajectory generation for different texts
3. Rendering pipeline
4. Parameter variation effects
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from simulator import (
    SimpleHandwritingSimulator,
    HandwritingRenderer,
    render_comparison,
)


def test_basic_generation():
    """Test basic trajectory generation."""
    print("\n" + "="*60)
    print("TEST 1: Basic trajectory generation")
    print("="*60)

    simulator = SimpleHandwritingSimulator(device='cpu')

    # Generate trajectory
    text = "hello"
    trajectory = simulator.generate(text, num_points=150, duration=2.0)

    print(f"Text: '{text}'")
    print(f"Trajectory shape: {trajectory.shape}")
    print(f"Position range: x=[{trajectory[:, 0].min():.2f}, {trajectory[:, 0].max():.2f}], "
          f"y=[{trajectory[:, 1].min():.2f}, {trajectory[:, 1].max():.2f}]")
    print(f"Pressure range: [{trajectory[:, 4].min():.2f}, {trajectory[:, 4].max():.2f}]")
    print(f"Velocity range: [{np.sqrt(trajectory[:, 2]**2 + trajectory[:, 3]**2).min():.2f}, "
          f"{np.sqrt(trajectory[:, 2]**2 + trajectory[:, 3]**2).max():.2f}]")

    # Render
    renderer = HandwritingRenderer()
    img = renderer.render_numpy(trajectory, normalize=True)

    # Save
    from PIL import Image
    Image.fromarray(img).save('/tmp/test_basic.png')
    print(f"✓ Saved to /tmp/test_basic.png")

    return trajectory


def test_different_texts():
    """Test generation for different texts."""
    print("\n" + "="*60)
    print("TEST 2: Different texts")
    print("="*60)

    simulator = SimpleHandwritingSimulator(device='cpu')
    texts = ["a", "hello", "abc123"]

    fig, axes = plt.subplots(1, len(texts), figsize=(12, 3))
    renderer = HandwritingRenderer(width=300, height=250)

    for idx, text in enumerate(texts):
        trajectory = simulator.generate(text, num_points=150)
        img = renderer.render_numpy(trajectory, normalize=True)

        axes[idx].imshow(img)
        axes[idx].set_title(f"'{text}'")
        axes[idx].axis('off')

        print(f"✓ Generated: '{text}' - shape {trajectory.shape}")

    plt.tight_layout()
    plt.savefig('/tmp/test_different_texts.png', dpi=100)
    print(f"✓ Saved to /tmp/test_different_texts.png")
    plt.close()


def test_parameter_variations():
    """Test how different parameters affect handwriting style."""
    print("\n" + "="*60)
    print("TEST 3: Parameter variations")
    print("="*60)

    simulator = SimpleHandwritingSimulator(device='cpu')
    renderer = HandwritingRenderer(width=250, height=200)

    # Get predefined styles
    styles = simulator.get_styles()
    text = "demo"

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for idx, (style_name, params) in enumerate(styles.items()):
        trajectory = simulator.generate(text, num_points=150, style_params=params)
        img = renderer.render_numpy(trajectory, normalize=True)

        axes[idx].imshow(img)
        axes[idx].set_title(f"Style: {style_name}")
        axes[idx].axis('off')

        print(f"✓ Generated style '{style_name}':")
        print(f"  - alpha (damping): {params.get('alpha', 'default')}")
        print(f"  - mu_p (pressure): {params.get('mu_p', 'default')}")

    plt.tight_layout()
    plt.savefig('/tmp/test_styles.png', dpi=100)
    print(f"✓ Saved to /tmp/test_styles.png")
    plt.close()


def test_trajectory_properties():
    """Test trajectory smoothness and physical properties."""
    print("\n" + "="*60)
    print("TEST 4: Trajectory properties")
    print("="*60)

    simulator = SimpleHandwritingSimulator(device='cpu')
    trajectory = simulator.generate("test", num_points=200)

    # Compute properties
    x, y = trajectory[:, 0], trajectory[:, 1]
    vx, vy = trajectory[:, 2], trajectory[:, 3]
    p = trajectory[:, 4]
    t = np.linspace(0, 2.0, len(trajectory))

    # Velocity
    speed = np.sqrt(vx**2 + vy**2)
    print(f"\nVelocity statistics:")
    print(f"  Mean speed: {speed.mean():.4f}")
    print(f"  Max speed: {speed.max():.4f}")

    # Acceleration (finite difference)
    ax = np.gradient(vx, t)
    ay = np.gradient(vy, t)
    accel = np.sqrt(ax**2 + ay**2)
    print(f"\nAcceleration statistics:")
    print(f"  Mean acceleration: {accel.mean():.4f}")
    print(f"  Max acceleration: {accel.max():.4f}")

    # Smoothness (second derivative)
    jerk = np.gradient(accel, t)
    print(f"\nJerk (3rd derivative - smoothness):")
    print(f"  Mean jerk: {jerk.mean():.4f}")
    print(f"  Max jerk: {jerk.max():.4f}")
    print(f"  (Lower jerk = smoother handwriting)")

    # Pressure
    print(f"\nPressure statistics:")
    print(f"  Mean pressure: {p.mean():.4f}")
    print(f"  Pressure range: [{p.min():.4f}, {p.max():.4f}]")

    # Plot trajectory properties
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Trajectory
    ax = axes[0, 0]
    ax.plot(x, y, 'b-', linewidth=1)
    ax.scatter(x[::10], y[::10], c=speed[::10], cmap='viridis', s=20)
    ax.set_aspect('equal')
    ax.set_title('Trajectory (colored by speed)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Speed over time
    ax = axes[0, 1]
    ax.plot(t, speed, 'b-', linewidth=1)
    ax.set_title('Velocity over time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Speed')
    ax.grid(True, alpha=0.3)

    # Acceleration over time
    ax = axes[1, 0]
    ax.plot(t, accel, 'g-', linewidth=1)
    ax.set_title('Acceleration over time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Acceleration')
    ax.grid(True, alpha=0.3)

    # Pressure over time
    ax = axes[1, 1]
    ax.plot(t, p, 'r-', linewidth=1)
    ax.set_title('Pressure over time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Pressure')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig('/tmp/test_trajectory_properties.png', dpi=100)
    print(f"\n✓ Saved to /tmp/test_trajectory_properties.png")
    plt.close()


def test_rendering():
    """Test rendering pipeline."""
    print("\n" + "="*60)
    print("TEST 5: Rendering pipeline")
    print("="*60)

    simulator = SimpleHandwritingSimulator(device='cpu')
    trajectory = simulator.generate("render", num_points=200)

    # Test PIL rendering
    renderer = HandwritingRenderer()
    img_pil = renderer.render_pil(trajectory, normalize=True)
    img_pil.save('/tmp/test_render_pil.png')
    print(f"✓ PIL rendering saved to /tmp/test_render_pil.png")

    # Test numpy rendering
    img_numpy = renderer.render_numpy(trajectory, normalize=True)
    print(f"✓ Numpy rendering: shape {img_numpy.shape}, dtype {img_numpy.dtype}")

    # Test matplotlib rendering (with analysis)
    fig = renderer.render_matplotlib(trajectory, normalize=True)
    fig.savefig('/tmp/test_render_matplotlib.png', dpi=100)
    print(f"✓ Matplotlib rendering saved to /tmp/test_render_matplotlib.png")
    plt.close()


def test_reproducibility():
    """Test that generation is reproducible with same seed."""
    print("\n" + "="*60)
    print("TEST 6: Reproducibility")
    print("="*60)

    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)

    simulator = SimpleHandwritingSimulator(device='cpu')
    traj1 = simulator.generate("test", num_points=100)

    # Reset seed
    torch.manual_seed(42)
    np.random.seed(42)

    traj2 = simulator.generate("test", num_points=100)

    # Should be identical
    diff = np.abs(traj1 - traj2).max()
    print(f"Max difference between two runs: {diff}")

    if diff < 1e-5:
        print("✓ Generation is reproducible")
    else:
        print("✗ Generation is NOT reproducible")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("PHASE 1 TEST SUITE: Handwriting ODE Simulator")
    print("="*60)

    try:
        test_basic_generation()
        test_different_texts()
        test_parameter_variations()
        test_trajectory_properties()
        test_rendering()
        test_reproducibility()

        print("\n" + "="*60)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nGenerated files:")
        print("  - /tmp/test_basic.png")
        print("  - /tmp/test_different_texts.png")
        print("  - /tmp/test_styles.png")
        print("  - /tmp/test_trajectory_properties.png")
        print("  - /tmp/test_render_pil.png")
        print("  - /tmp/test_render_matplotlib.png")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

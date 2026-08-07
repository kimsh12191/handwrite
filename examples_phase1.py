#!/usr/bin/env python3
"""
Phase 1 Quick Start Examples
간단한 사용 예시
"""

import sys
sys.path.insert(0, '/home/user/handwrite')

from simulator import SimpleHandwritingSimulator, HandwritingRenderer
import numpy as np
from PIL import Image


def example_1_basic():
    """Example 1: 기본 사용법"""
    print("\n" + "="*60)
    print("Example 1: 기본 손글씨 생성")
    print("="*60)

    # 시뮬레이터 생성
    simulator = SimpleHandwritingSimulator(device='cpu')

    # 손글씨 생성
    text = "hello"
    trajectory = simulator.generate(
        text=text,
        num_points=150,  # 시간 스텝
        duration=2.0      # 지속시간
    )

    print(f"✓ 생성됨: '{text}'")
    print(f"  - 궤적 크기: {trajectory.shape}")
    print(f"  - 위치: x=[{trajectory[:, 0].min():.2f}, {trajectory[:, 0].max():.2f}]")
    print(f"         y=[{trajectory[:, 1].min():.2f}, {trajectory[:, 1].max():.2f}]")
    print(f"  - 필압: [{trajectory[:, 4].min():.2f}, {trajectory[:, 4].max():.2f}]")

    # 렌더링
    renderer = HandwritingRenderer(width=400, height=300)
    img = renderer.render_pil(trajectory, normalize=True)

    # 저장
    img.save('/tmp/example1_hello.png')
    print(f"✓ 저장: /tmp/example1_hello.png")

    return trajectory, img


def example_2_different_texts():
    """Example 2: 다양한 텍스트"""
    print("\n" + "="*60)
    print("Example 2: 다양한 텍스트 생성")
    print("="*60)

    simulator = SimpleHandwritingSimulator(device='cpu')
    renderer = HandwritingRenderer(width=300, height=250)

    texts = ["a", "hi", "test", "123"]

    for text in texts:
        trajectory = simulator.generate(text, num_points=120)
        img = renderer.render_pil(trajectory, normalize=True)
        filename = f'/tmp/example2_{text}.png'
        img.save(filename)
        print(f"✓ '{text}' → {filename}")


def example_3_styles():
    """Example 3: 스타일 변화"""
    print("\n" + "="*60)
    print("Example 3: 다양한 필체 스타일")
    print("="*60)

    simulator = SimpleHandwritingSimulator(device='cpu')
    renderer = HandwritingRenderer(width=300, height=250)

    text = "style"
    styles = simulator.get_styles()

    for style_name in styles.keys():
        trajectory = simulator.generate_with_style(text, style_name=style_name)
        img = renderer.render_pil(trajectory, normalize=True)
        filename = f'/tmp/example3_{style_name}.png'
        img.save(filename)
        print(f"✓ Style '{style_name}' → {filename}")


def example_4_custom_parameters():
    """Example 4: 커스텀 파라미터"""
    print("\n" + "="*60)
    print("Example 4: 커스텀 파라미터로 필체 조절")
    print("="*60)

    simulator = SimpleHandwritingSimulator(device='cpu')
    renderer = HandwritingRenderer(width=300, height=250)

    text = "custom"

    # 커스텀 파라미터 세트
    param_sets = {
        'smooth': {'alpha': 0.45, 'gamma': 0.02, 'mu_p': 0.25},
        'rough': {'alpha': 0.2, 'gamma': 0.12, 'mu_p': 0.45},
        'relaxed': {'alpha': 0.35, 'beta': 0.06, 'mu_p': 0.2},
        'fast': {'alpha': 0.5, 'gamma': 0.08, 'mu_p': 0.35},
    }

    for param_name, params in param_sets.items():
        trajectory = simulator.generate(text, num_points=150, style_params=params)
        img = renderer.render_pil(trajectory, normalize=True)
        filename = f'/tmp/example4_{param_name}.png'
        img.save(filename)
        print(f"✓ '{param_name}' → {filename}")
        print(f"  파라미터: {params}")


def example_5_trajectory_analysis():
    """Example 5: 궤적 분석"""
    print("\n" + "="*60)
    print("Example 5: 궤적의 물리 특성 분석")
    print("="*60)

    simulator = SimpleHandwritingSimulator(device='cpu')
    trajectory = simulator.generate("analyze", num_points=200, duration=2.0)

    # 각 차원 추출
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    vx = trajectory[:, 2]
    vy = trajectory[:, 3]
    p = trajectory[:, 4]
    theta = trajectory[:, 5]

    # 시간 축
    t = np.linspace(0, 2.0, len(trajectory))

    # 기본 통계
    print(f"\n위치 (Position):")
    print(f"  X: [{x.min():.3f}, {x.max():.3f}] (범위: {x.max()-x.min():.3f})")
    print(f"  Y: [{y.min():.3f}, {y.max():.3f}] (범위: {y.max()-y.min():.3f})")

    # 속도
    speed = np.sqrt(vx**2 + vy**2)
    print(f"\n속도 (Velocity):")
    print(f"  Mean: {speed.mean():.4f}")
    print(f"  Max: {speed.max():.4f}")
    print(f"  Min: {speed.min():.4f}")

    # 가속도 (유한 차분)
    ax = np.gradient(vx, t)
    ay = np.gradient(vy, t)
    accel = np.sqrt(ax**2 + ay**2)
    print(f"\n가속도 (Acceleration):")
    print(f"  Mean: {accel.mean():.4f}")
    print(f"  Max: {accel.max():.4f}")

    # 저크 (smoothness = 3차 미분)
    jerk = np.gradient(accel, t)
    print(f"\n저크 (3차 미분 - 부드러움 정도):")
    print(f"  Mean: {jerk.mean():.4f}")
    print(f"  Max: {jerk.max():.4f}")
    print(f"  (낮을수록 더 부드러운 필체)")

    # 필압
    print(f"\n필압 (Pressure):")
    print(f"  Mean: {p.mean():.4f}")
    print(f"  Max: {p.max():.4f}")
    print(f"  Min (touching): {(p > 0.05).sum() / len(p) * 100:.1f}% 접촉")

    # 각도
    print(f"\n펜 각도 (Pen Angle):")
    print(f"  Mean: {theta.mean():.4f}")
    print(f"  Std: {theta.std():.4f}")

    print(f"\n✓ 분석 완료")


def example_6_comparison():
    """Example 6: 스타일 비교"""
    print("\n" + "="*60)
    print("Example 6: 같은 텍스트, 다른 스타일 비교")
    print("="*60)

    simulator = SimpleHandwritingSimulator(device='cpu')
    renderer = HandwritingRenderer(width=250, height=200)

    text = "compare"
    styles = ['normal', 'shaky', 'flowing', 'heavy']

    # 한 번에 모든 스타일 생성
    trajectories = {}
    images = {}

    print(f"\n텍스트: '{text}'")
    print(f"\n스타일별 생성:")

    for style in styles:
        traj = simulator.generate_with_style(text, style_name=style)
        img = renderer.render_pil(traj, normalize=True)

        trajectories[style] = traj
        images[style] = img

        # 분석
        speed = np.sqrt(traj[:, 2]**2 + traj[:, 3]**2)
        print(f"  {style:10} - 평균속도: {speed.mean():.4f}, "
              f"필압: {traj[:, 4].mean():.4f}")

    # 이미지 결합 (2x2 그리드)
    grid_width = 500
    grid_height = 400
    grid_img = Image.new('RGB', (grid_width, grid_height), 'white')

    cell_width = grid_width // 2
    cell_height = grid_height // 2

    positions = [
        ((0, 0), 'normal'),
        ((cell_width, 0), 'shaky'),
        ((0, cell_height), 'flowing'),
        ((cell_width, cell_height), 'heavy'),
    ]

    for (x, y), style in positions:
        # 셀 크기에 맞게 리사이즈
        img = images[style].resize((cell_width, cell_height))
        grid_img.paste(img, (x, y))

    grid_img.save('/tmp/example6_styles_grid.png')
    print(f"\n✓ 비교 그리드 저장: /tmp/example6_styles_grid.png")


if __name__ == '__main__':
    try:
        print("\n" + "="*60)
        print("Phase 1 Quick Start Examples")
        print("="*60)

        example_1_basic()
        example_2_different_texts()
        example_3_styles()
        example_4_custom_parameters()
        example_5_trajectory_analysis()
        example_6_comparison()

        print("\n" + "="*60)
        print("✓ 모든 예시 완료!")
        print("="*60)
        print("\n생성된 파일:")
        import os
        files = [f for f in os.listdir('/tmp') if f.startswith('example')]
        for f in sorted(files):
            print(f"  /tmp/{f}")

    except Exception as e:
        print(f"\n✗ 오류: {e}")
        import traceback
        traceback.print_exc()

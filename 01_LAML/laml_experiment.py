"""
LAML: Lagrangian Action Minimization Learning
==============================================

완전히 새로운 학습 패러다임 구현

핵심 아이디어:
1. 데이터 → 최종 가중치 예측 (메타 예측)
2. 시작 → 끝 최적 궤적 계산 (BVP)
3. 최소 작용 원리 만족 여부 검증
4. 불만족 시 탐색 및 보정
5. 강화학습식 랜덤성과 자기확신 조절
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import time


class LightweightNN:
    """초경량 신경망 (4→6→1)"""

    def __init__(self, seed=None):
        if seed:
            np.random.seed(seed)

        # 작은 가중치로 초기화
        self.W1 = np.random.randn(4, 6) * 0.1
        self.b1 = np.zeros(6)
        self.W2 = np.random.randn(6, 1) * 0.1
        self.b2 = np.zeros(1)

    def forward(self, X):
        """순전파: ReLU 활성화"""
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def loss(self, X, y):
        """MSE 손실"""
        pred = self.forward(X)
        return np.mean((pred - y) ** 2)

    def get_weights(self):
        """모든 가중치를 1D 벡터로"""
        return np.concatenate([
            self.W1.flatten(), self.b1,
            self.W2.flatten(), self.b2
        ])

    def set_weights(self, w):
        """1D 벡터에서 가중치 복원"""
        self.W1 = w[:24].reshape(4, 6)
        self.b1 = w[24:30]
        self.W2 = w[30:36].reshape(6, 1)
        self.b2 = w[36:37]

    def gradient(self, X, y):
        """역전파로 그래디언트 계산"""
        m = len(X)
        pred = self.forward(X)

        # 역전파
        dz2 = 2 * (pred - y) / m
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        return np.concatenate([
            dW1.flatten(), db1,
            dW2.flatten(), db2
        ])


class MetaPredictor:
    """
    메타 예측기: 데이터 → 최종 가중치 예측

    실제로는 많은 학습 경험으로부터 학습된 모델이어야 하지만,
    여기서는 간단한 휴리스틱 사용
    """

    def predict(self, X, y, current_w, network):
        """
        데이터 특성으로부터 최종 가중치를 예측

        전략:
        1. 현재 gradient 방향 계산
        2. 데이터 분산을 고려한 step size
        3. 약간의 랜덤 탐색
        """
        # Gradient 방향
        grad = network.gradient(X, y)

        # 데이터 특성 기반 스케일
        y_std = np.std(y)
        scale = 2.0 * (1 + y_std)

        # 예측: 현재 + (gradient 반대 방향) + 탐색
        predicted = current_w - scale * grad
        predicted += np.random.randn(len(current_w)) * 0.3

        return predicted


class BVPSolver:
    """
    경계값 문제(BVP) 솔버: θ₀ → θ_T 최적 궤적 계산

    이상적으로는 Euler-Lagrange 방정식을 풀어야 하지만,
    여기서는 smoothstep 보간 사용 (계산 효율성)
    """

    def solve(self, theta_0, theta_T, steps=10):
        """Smoothstep 보간으로 부드러운 궤적 생성"""
        trajectory = []

        for i in range(steps):
            t = i / (steps - 1)
            # Smoothstep: 3t² - 2t³
            smooth = 3 * t**2 - 2 * t**3
            theta_t = theta_0 + smooth * (theta_T - theta_0)
            trajectory.append(theta_t)

        return np.array(trajectory)


class ActionCalculator:
    """
    작용(Action) 계산기

    S = ∫[½||θ̇||² + λL(θ)] dt

    - 운동 에너지: ½||θ̇||²  (변화의 빠르기)
    - 포텐셜: λL(θ)        (손실 함수)
    """

    def __init__(self, network, X, y, lambda_loss=1.0):
        self.net = network
        self.X = X
        self.y = y
        self.lambda_loss = lambda_loss

    def compute(self, trajectory):
        """궤적을 따라 작용 적분"""
        action = 0.0

        for i in range(len(trajectory) - 1):
            theta_t = trajectory[i]
            theta_next = trajectory[i + 1]

            # 속도: θ̇
            velocity = theta_next - theta_t
            kinetic = 0.5 * np.sum(velocity ** 2)

            # 손실
            self.net.set_weights(theta_t)
            loss = self.net.loss(self.X, self.y)
            potential = self.lambda_loss * loss

            # Lagrangian
            action += kinetic + potential

        return action / len(trajectory)


class LAMLOptimizer:
    """
    LAML 최적화기: 새로운 학습 패러다임

    알고리즘:
    1. 데이터 → 최종 가중치 예측
    2. 시작 → 끝 궤적 계산
    3. Action 검증
    4. 불만족 → 탐색 & 보정
    5. 만족 → 업데이트
    6. 랜덤성 + 자기확신 조절
    """

    def __init__(self, network,
                 action_threshold=0.5,
                 learning_rate=0.1,
                 explore_samples=8):
        self.net = network
        self.action_threshold = action_threshold
        self.lr = learning_rate
        self.explore_samples = explore_samples

        self.meta_predictor = MetaPredictor()
        self.bvp_solver = BVPSolver()

        # 추적
        self.history = {
            'loss': [],
            'action': [],
            'confidence': [],
            'accept_rate': []
        }
        self.confidence = 1.0  # 자기확신
        self.accepts = 0
        self.rejects = 0

    def train(self, X, y, max_iters=100, verbose=True):
        """LAML 학습 루프"""
        start = time.time()
        action_calc = ActionCalculator(self.net, X, y)

        for it in range(max_iters):
            # 1. 메타 예측: 데이터 → 최종 가중치
            theta_0 = self.net.get_weights()
            theta_pred = self.meta_predictor.predict(X, y, theta_0, self.net)

            # 2. BVP: 시작 → 끝 궤적
            traj = self.bvp_solver.solve(theta_0, theta_pred, steps=10)

            # 3. Action 계산
            action = action_calc.compute(traj)
            loss_before = self.net.loss(X, y)

            # 4. Action이 너무 크면 탐색 & 보정
            if action > self.action_threshold:
                theta_pred = self._explore_alternatives(
                    X, y, theta_0, theta_pred, action_calc
                )
                traj = self.bvp_solver.solve(theta_0, theta_pred, steps=10)
                action = action_calc.compute(traj)
                self.rejects += 1
            else:
                self.accepts += 1

            # 5. 궤적을 따라 업데이트
            step_size = self.lr * self.confidence
            direction = traj[1] - theta_0

            # 랜덤 탐색 추가 (강화학습 효과)
            noise = np.random.randn(len(theta_0)) * 0.02 * self.confidence
            theta_new = theta_0 + step_size * direction + noise

            self.net.set_weights(theta_new)
            loss_after = self.net.loss(X, y)

            # 6. 자기확신 조절
            if loss_after < loss_before:
                self.confidence = min(1.0, self.confidence * 1.05)
            else:
                self.confidence *= 0.95
                self.confidence = max(0.1, self.confidence)

            # 추적
            self.history['loss'].append(loss_after)
            self.history['action'].append(action)
            self.history['confidence'].append(self.confidence)
            accept_rate = self.accepts / (self.accepts + self.rejects + 1e-8)
            self.history['accept_rate'].append(accept_rate)

            if verbose and it % 10 == 0:
                print(f"[{it:3d}] Loss: {loss_after:.5f} | "
                      f"Action: {action:.4f} | "
                      f"Conf: {self.confidence:.3f} | "
                      f"Accept: {accept_rate:.2%}")

            # 조기 종료
            if action < self.action_threshold and loss_after < 0.01:
                if verbose:
                    print(f"\n✓ Converged at iteration {it}")
                break

        elapsed = time.time() - start
        return {
            'final_loss': self.history['loss'][-1],
            'iterations': len(self.history['loss']),
            'time': elapsed
        }

    def _explore_alternatives(self, X, y, theta_0, theta_current, action_calc):
        """
        Action이 높을 때 대안 탐색
        강화학습식 랜덤 탐색
        """
        best_theta = theta_current
        best_action = float('inf')

        for _ in range(self.explore_samples):
            # 랜덤 섭동
            noise = np.random.randn(len(theta_current)) * 0.3 * self.confidence
            candidate = theta_current + noise

            # 평가
            traj = self.bvp_solver.solve(theta_0, candidate, steps=10)
            action = action_calc.compute(traj)

            if action < best_action:
                best_action = action
                best_theta = candidate

        return best_theta


class SGDOptimizer:
    """비교를 위한 표준 SGD"""

    def __init__(self, network, learning_rate=0.1):
        self.net = network
        self.lr = learning_rate
        self.history = {'loss': []}

    def train(self, X, y, max_iters=100, verbose=False):
        """표준 gradient descent"""
        start = time.time()

        for it in range(max_iters):
            grad = self.net.gradient(X, y)
            w = self.net.get_weights()
            w -= self.lr * grad
            self.net.set_weights(w)

            loss = self.net.loss(X, y)
            self.history['loss'].append(loss)

            if verbose and it % 10 == 0:
                print(f"[{it:3d}] Loss: {loss:.5f}")

            if loss < 0.01:
                break

        elapsed = time.time() - start
        return {
            'final_loss': self.history['loss'][-1],
            'iterations': len(self.history['loss']),
            'time': elapsed
        }


def make_dataset(name='nonlinear', n=100):
    """테스트 데이터셋 생성"""
    np.random.seed(42)
    X = np.random.randn(n, 4)

    if name == 'linear':
        y = (X @ [1, -0.5, 0.3, 0.8]).reshape(-1, 1)
    elif name == 'nonlinear':
        y = (np.sin(X[:, 0]) + np.cos(X[:, 1]) * X[:, 2]).reshape(-1, 1)
    elif name == 'xor':
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(float).reshape(-1, 1)

    # 정규화
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)

    return X, y


def run_experiment(dataset_name='nonlinear'):
    """실험 실행: LAML vs SGD"""
    print(f"\n{'='*70}")
    print(f"실험: {dataset_name.upper()} 데이터셋")
    print(f"{'='*70}\n")

    X, y = make_dataset(dataset_name, n=100)
    print(f"데이터: X={X.shape}, y={y.shape}\n")

    # LAML
    print("1️⃣  LAML (Lagrangian Action Minimization Learning)")
    print("-" * 70)
    net_laml = LightweightNN(seed=42)
    opt_laml = LAMLOptimizer(net_laml, action_threshold=0.5, learning_rate=0.1)
    result_laml = opt_laml.train(X, y, max_iters=100, verbose=True)

    # SGD
    print(f"\n2️⃣  Standard SGD")
    print("-" * 70)
    net_sgd = LightweightNN(seed=42)
    opt_sgd = SGDOptimizer(net_sgd, learning_rate=0.1)
    result_sgd = opt_sgd.train(X, y, max_iters=100, verbose=True)

    # 비교
    print(f"\n{'='*70}")
    print("📊 결과 비교")
    print(f"{'='*70}")
    print(f"{'지표':<25} {'LAML':>20} {'SGD':>20}")
    print("-" * 70)
    print(f"{'최종 손실':<25} {result_laml['final_loss']:>20.6f} {result_sgd['final_loss']:>20.6f}")
    print(f"{'수렴 반복 횟수':<25} {result_laml['iterations']:>20d} {result_sgd['iterations']:>20d}")
    print(f"{'학습 시간 (초)':<25} {result_laml['time']:>20.4f} {result_sgd['time']:>20.4f}")

    improvement = (result_sgd['final_loss'] - result_laml['final_loss']) / result_sgd['final_loss'] * 100
    print(f"{'손실 개선율 (%)':<25} {improvement:>20.2f}")
    print("="*70)

    return {
        'laml': result_laml,
        'sgd': result_sgd,
        'laml_opt': opt_laml,
        'sgd_opt': opt_sgd,
        'dataset': dataset_name
    }


def plot_comparison(results):
    """결과 시각화"""
    laml_opt = results['laml_opt']
    sgd_opt = results['sgd_opt']
    dataset = results['dataset']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'LAML vs SGD: {dataset.upper()} Dataset',
                 fontsize=16, fontweight='bold')

    # 1. 손실 곡선
    ax = axes[0, 0]
    ax.plot(laml_opt.history['loss'], 'b-', label='LAML', linewidth=2, alpha=0.8)
    ax.plot(sgd_opt.history['loss'], 'r--', label='SGD', linewidth=2, alpha=0.8)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('학습 곡선', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 2. Action (LAML만)
    ax = axes[0, 1]
    ax.plot(laml_opt.history['action'], 'g-', linewidth=2, alpha=0.8)
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Action S', fontsize=12)
    ax.set_title('작용 함수 (LAML)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 3. 자기확신
    ax = axes[1, 0]
    ax.plot(laml_opt.history['confidence'], 'm-', linewidth=2, alpha=0.8)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Confidence', fontsize=12)
    ax.set_title('자기확신 변화 (LAML)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])

    # 4. 수락률
    ax = axes[1, 1]
    ax.plot(laml_opt.history['accept_rate'], 'c-', linewidth=2, alpha=0.8)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Accept Rate', fontsize=12)
    ax.set_title('예측 수락률 (LAML)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])

    plt.tight_layout()

    filename = f'/Users/say/Documents/GitHub/ai/laml_{dataset}_results.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✅ 시각화 저장: {filename}")

    return fig


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 LAML: Lagrangian Action Minimization Learning")
    print("   물리학에서 영감을 받은 완전히 새로운 학습 패러다임")
    print("="*70)

    # 여러 데이터셋에서 실험
    datasets = ['linear', 'nonlinear', 'xor']
    all_results = {}

    for ds in datasets:
        result = run_experiment(ds)
        all_results[ds] = result
        plot_comparison(result)

    # 최종 요약
    print(f"\n{'='*70}")
    print("🎯 전체 요약")
    print(f"{'='*70}")

    for ds, res in all_results.items():
        laml_loss = res['laml']['final_loss']
        sgd_loss = res['sgd']['final_loss']
        improvement = (sgd_loss - laml_loss) / sgd_loss * 100

        if improvement > 0:
            status = "✅ LAML 승리"
        else:
            status = "❌ SGD 승리"

        print(f"{ds.upper():12s} | {status:15s} | 개선: {improvement:+.2f}%")

    print("\n" + "="*70)
    print("✅ 실험 완료!")
    print("="*70 + "\n")

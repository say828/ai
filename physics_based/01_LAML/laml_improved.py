"""
LAML v2.0: Improved with Gradient-Based Prediction
===================================================

개선사항:
1. 메타 예측 → Gradient 기반 예측으로 완화
2. Action을 직접 최소화하는 방향 탐색
3. 적응형 step size
"""

import numpy as np
import matplotlib.pyplot as plt
import time


class LightweightNN:
    """경량 신경망"""

    def __init__(self, seed=None):
        if seed:
            np.random.seed(seed)
        self.W1 = np.random.randn(4, 6) * 0.1
        self.b1 = np.zeros(6)
        self.W2 = np.random.randn(6, 1) * 0.1
        self.b2 = np.zeros(1)

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def loss(self, X, y):
        pred = self.forward(X)
        return np.mean((pred - y) ** 2)

    def get_weights(self):
        return np.concatenate([
            self.W1.flatten(), self.b1,
            self.W2.flatten(), self.b2
        ])

    def set_weights(self, w):
        self.W1 = w[:24].reshape(4, 6)
        self.b1 = w[24:30]
        self.W2 = w[30:36].reshape(6, 1)
        self.b2 = w[36:37]

    def gradient(self, X, y):
        m = len(X)
        pred = self.forward(X)
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


class ImprovedPredictor:
    """
    개선된 예측기: Gradient 기반

    핵심 아이디어:
    - 메타 예측 대신, gradient 방향으로 여러 step 예측
    - "어디까지 가야 수렴하는가"를 추정
    """

    def predict_endpoint(self, network, X, y, current_w, num_lookahead=5):
        """
        Gradient 방향으로 lookahead하여 예상 종착지 추정
        """
        theta = current_w.copy()

        for _ in range(num_lookahead):
            grad = network.gradient(X, y)
            theta -= 0.1 * grad  # 가상의 업데이트

        return theta


def compute_action_simple(theta_0, theta_T, network, X, y, lambda_loss=1.0):
    """
    간소화된 Action 계산

    S ≈ ||θ_T - θ_0||² + λ * Loss(θ_T)

    궤적을 따라 적분하는 대신, 시작과 끝만 사용
    """
    distance = np.sum((theta_T - theta_0) ** 2)
    network.set_weights(theta_T)
    loss = network.loss(X, y)
    return distance + lambda_loss * loss


class LAMLv2Optimizer:
    """
    LAML v2: 개선된 버전

    변경사항:
    1. Gradient lookahead로 종착지 예측
    2. 간소화된 Action 계산
    3. Action 최소화를 목표로 방향 탐색
    """

    def __init__(self, network, lambda_loss=1.0, learning_rate=0.1):
        self.net = network
        self.lambda_loss = lambda_loss
        self.lr = learning_rate
        self.predictor = ImprovedPredictor()

        self.history = {
            'loss': [],
            'action': [],
            'step_size': []
        }

    def train(self, X, y, max_iters=100, verbose=True):
        start = time.time()

        for it in range(max_iters):
            theta_0 = self.net.get_weights()

            # 1. Gradient lookahead로 종착지 예측
            theta_pred = self.predictor.predict_endpoint(
                self.net, X, y, theta_0, num_lookahead=5
            )

            # 2. Action 계산
            action = compute_action_simple(
                theta_0, theta_pred, self.net, X, y, self.lambda_loss
            )

            # 3. Action을 최소화하는 방향 탐색
            best_theta = theta_pred
            best_action = action

            # 여러 방향 시도
            for scale in [0.5, 1.0, 2.0]:
                direction = theta_pred - theta_0
                candidate = theta_0 + scale * direction

                cand_action = compute_action_simple(
                    theta_0, candidate, self.net, X, y, self.lambda_loss
                )

                if cand_action < best_action:
                    best_action = cand_action
                    best_theta = candidate

            # 4. 업데이트
            step_size = self.lr
            theta_new = theta_0 + step_size * (best_theta - theta_0)
            self.net.set_weights(theta_new)

            loss = self.net.loss(X, y)
            self.history['loss'].append(loss)
            self.history['action'].append(best_action)
            self.history['step_size'].append(step_size)

            if verbose and it % 10 == 0:
                print(f"[{it:3d}] Loss: {loss:.5f} | Action: {best_action:.4f}")

            if loss < 0.01:
                if verbose:
                    print(f"\n✓ Converged at iteration {it}")
                break

        elapsed = time.time() - start
        return {
            'final_loss': self.history['loss'][-1],
            'iterations': len(self.history['loss']),
            'time': elapsed
        }


class SGDOptimizer:
    """표준 SGD"""

    def __init__(self, network, learning_rate=0.1):
        self.net = network
        self.lr = learning_rate
        self.history = {'loss': []}

    def train(self, X, y, max_iters=100, verbose=False):
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
    """데이터셋 생성"""
    np.random.seed(42)
    X = np.random.randn(n, 4)
    if name == 'linear':
        y = (X @ [1, -0.5, 0.3, 0.8]).reshape(-1, 1)
    elif name == 'nonlinear':
        y = (np.sin(X[:, 0]) + np.cos(X[:, 1]) * X[:, 2]).reshape(-1, 1)
    elif name == 'xor':
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(float).reshape(-1, 1)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)
    return X, y


def run_improved_experiment(dataset_name='nonlinear'):
    """개선된 실험"""
    print(f"\n{'='*70}")
    print(f"개선된 실험: {dataset_name.upper()}")
    print(f"{'='*70}\n")

    X, y = make_dataset(dataset_name, n=100)

    # LAML v2
    print("1️⃣  LAML v2.0 (Gradient-Based)")
    print("-" * 70)
    net_laml = LightweightNN(seed=42)
    opt_laml = LAMLv2Optimizer(net_laml, lambda_loss=1.0, learning_rate=0.1)
    result_laml = opt_laml.train(X, y, max_iters=100, verbose=True)

    # SGD
    print(f"\n2️⃣  Standard SGD")
    print("-" * 70)
    net_sgd = LightweightNN(seed=42)
    opt_sgd = SGDOptimizer(net_sgd, learning_rate=0.1)
    result_sgd = opt_sgd.train(X, y, max_iters=100, verbose=True)

    # 비교
    print(f"\n{'='*70}")
    print("📊 결과")
    print(f"{'='*70}")
    print(f"{'지표':<25} {'LAML v2':>20} {'SGD':>20}")
    print("-" * 70)
    print(f"{'최종 손실':<25} {result_laml['final_loss']:>20.6f} {result_sgd['final_loss']:>20.6f}")
    print(f"{'수렴 반복 횟수':<25} {result_laml['iterations']:>20d} {result_sgd['iterations']:>20d}")
    print(f"{'시간 (초)':<25} {result_laml['time']:>20.4f} {result_sgd['time']:>20.4f}")

    improvement = (result_sgd['final_loss'] - result_laml['final_loss']) / result_sgd['final_loss'] * 100
    print(f"{'손실 개선율':<25} {improvement:>20.2f}%")
    print("="*70)

    return {
        'laml': result_laml,
        'sgd': result_sgd,
        'laml_opt': opt_laml,
        'sgd_opt': opt_sgd,
        'dataset': dataset_name
    }


def plot_improved(results):
    """시각화"""
    laml = results['laml_opt']
    sgd = results['sgd_opt']
    ds = results['dataset']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'LAML v2 vs SGD: {ds.upper()}', fontsize=16, fontweight='bold')

    # Loss
    ax = axes[0]
    ax.plot(laml.history['loss'], 'b-', label='LAML v2', linewidth=2)
    ax.plot(sgd.history['loss'], 'r--', label='SGD', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Learning Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Action
    ax = axes[1]
    ax.plot(laml.history['action'], 'g-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Action')
    ax.set_title('Action over Time')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'/Users/say/Documents/GitHub/ai/laml_v2_{ds}_results.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✅ 저장: {filename}")


if __name__ == "__main__":
    print("="*70)
    print("🚀 LAML v2.0: Gradient-Based Improvement")
    print("="*70)

    datasets = ['linear', 'nonlinear', 'xor']
    all_results = {}

    for ds in datasets:
        result = run_improved_experiment(ds)
        all_results[ds] = result
        plot_improved(result)

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

        print(f"{ds.upper():12s} | {status:15s} | {improvement:+.2f}%")

    print("\n" + "="*70)

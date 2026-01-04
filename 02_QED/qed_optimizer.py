"""
QED: Quantum-Inspired Ensemble Descent
========================================

완전히 새로운 최적화 패러다임

영감:
- 양자역학: 여러 상태를 동시에 탐색 (중첩)
- 진화론: 좋은 해는 생존, 나쁜 해는 변이
- 집단 지성: 여러 agent가 협력하여 최적해 탐색
- 열역학: 온도로 탐색-수렴 조절
- 신경과학: 성공한 경로 강화

핵심 메커니즘:
1. N개의 "입자"(가중치)를 동시에 유지
2. 각 입자는 독립적으로 gradient descent
3. 주기적으로 집단 정보 공유 (평균, 최선)
4. 성능 나쁜 입자는 "돌연변이" 또는 "재생성"
5. 양자 터널링: 확률적으로 나쁜 방향도 시도
6. 온도 감소: 탐색 → 수렴
7. 최종: 앙상블 결합
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple


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

    def copy(self):
        new = LightweightNN()
        new.set_weights(self.get_weights().copy())
        return new


class Particle:
    """
    입자: 가중치 공간을 탐색하는 agent

    개념:
    - 위치: 현재 가중치
    - 속도: 업데이트 방향 (momentum)
    - 에너지: 손실값 (낮을수록 좋음)
    - 온도: 탐색 정도 (높으면 랜덤 탐색 강화)
    """

    def __init__(self, network: LightweightNN, particle_id: int):
        self.network = network
        self.id = particle_id
        self.velocity = np.zeros_like(network.get_weights())
        self.best_position = network.get_weights().copy()
        self.best_loss = float('inf')

    def update(self, X, y, learning_rate: float, temperature: float,
               global_best: np.ndarray, swarm_center: np.ndarray):
        """
        입자 업데이트

        영감:
        - Gradient descent: 경사 하강
        - Momentum: 관성
        - PSO: 자기 최선 + 집단 최선
        - 양자 터널링: 확률적 점프
        """
        # 1. Gradient 계산
        grad = self.network.gradient(X, y)

        # 2. 여러 힘의 조합
        current_pos = self.network.get_weights()

        # Force 1: Gradient (기본 학습)
        force_gradient = -grad

        # Force 2: Momentum (관성)
        force_momentum = 0.9 * self.velocity

        # Force 3: Attraction to personal best (자기 최선)
        force_personal = 0.5 * (self.best_position - current_pos)

        # Force 4: Attraction to global best (집단 최선)
        force_global = 0.3 * (global_best - current_pos)

        # Force 5: Attraction to swarm center (집단 중심)
        force_center = 0.2 * (swarm_center - current_pos)

        # Force 6: Quantum tunneling (양자 터널링)
        force_quantum = np.random.randn(len(current_pos)) * temperature

        # 총 힘
        total_force = (
            force_gradient +
            force_momentum +
            force_personal +
            force_global +
            force_center +
            force_quantum
        )

        # 3. 속도 업데이트
        self.velocity = total_force

        # 4. 위치 업데이트
        new_position = current_pos + learning_rate * self.velocity
        self.network.set_weights(new_position)

        # 5. 평가
        current_loss = self.network.loss(X, y)

        # 6. 개인 최선 업데이트
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_position = new_position.copy()

        return current_loss


class QEDOptimizer:
    """
    QED: Quantum-Inspired Ensemble Descent

    핵심:
    - N개의 입자가 협력하여 최적해 탐색
    - 양자 중첩 모방: 여러 상태 동시 유지
    - 진화적 선택: 나쁜 입자는 재생성/변이
    - 온도 조절: 탐색 → 수렴
    """

    def __init__(self,
                 network_template: LightweightNN,
                 n_particles: int = 10,
                 learning_rate: float = 0.1,
                 temperature_init: float = 0.5,
                 temperature_decay: float = 0.95):
        self.n_particles = n_particles
        self.lr = learning_rate
        self.temperature = temperature_init
        self.temp_decay = temperature_decay

        # 입자 초기화
        self.particles: List[Particle] = []
        for i in range(n_particles):
            net = network_template.copy()
            # 각 입자를 약간 다르게 초기화
            noise = np.random.randn(len(net.get_weights())) * 0.1
            net.set_weights(net.get_weights() + noise)
            particle = Particle(net, i)
            self.particles.append(particle)

        # 전역 최선
        self.global_best = network_template.get_weights().copy()
        self.global_best_loss = float('inf')

        # 추적
        self.history = {
            'loss': [],
            'best_loss': [],
            'diversity': [],
            'temperature': []
        }

    def get_swarm_center(self) -> np.ndarray:
        """집단의 중심 (평균 위치)"""
        positions = [p.network.get_weights() for p in self.particles]
        return np.mean(positions, axis=0)

    def get_diversity(self) -> float:
        """집단의 다양성 (분산)"""
        positions = [p.network.get_weights() for p in self.particles]
        return np.std(positions)

    def evolve(self, X, y):
        """
        진화적 선택: 나쁜 입자는 변이/재생성

        전략:
        1. 하위 30% 입자 선택
        2. Crossover: 좋은 입자 2개 결합
        3. Mutation: 랜덤 변이
        """
        # 성능으로 정렬
        losses = [(p.network.loss(X, y), i) for i, p in enumerate(self.particles)]
        losses.sort()

        # 하위 30%
        n_replace = max(1, self.n_particles // 3)
        worst_indices = [idx for _, idx in losses[-n_replace:]]
        best_indices = [idx for _, idx in losses[:3]]

        for i in worst_indices:
            # 전략 1: Crossover (80% 확률)
            if np.random.rand() < 0.8:
                # 좋은 입자 2개 선택
                parent1 = self.particles[np.random.choice(best_indices)]
                parent2 = self.particles[np.random.choice(best_indices)]

                # 결합 (평균 + 약간의 변이)
                w1 = parent1.network.get_weights()
                w2 = parent2.network.get_weights()
                child_w = 0.5 * (w1 + w2) + np.random.randn(len(w1)) * 0.1

                self.particles[i].network.set_weights(child_w)
                self.particles[i].velocity *= 0  # 속도 리셋

            # 전략 2: Mutation (20% 확률)
            else:
                # 전역 최선에서 출발 + 큰 변이
                mutated = self.global_best + np.random.randn(len(self.global_best)) * 0.3
                self.particles[i].network.set_weights(mutated)
                self.particles[i].velocity *= 0

    def train(self, X, y, max_iters: int = 100, verbose: bool = True):
        """QED 학습"""
        start = time.time()

        for iteration in range(max_iters):
            swarm_center = self.get_swarm_center()
            losses = []

            # 각 입자 업데이트
            for particle in self.particles:
                loss = particle.update(
                    X, y,
                    self.lr,
                    self.temperature,
                    self.global_best,
                    swarm_center
                )
                losses.append(loss)

                # 전역 최선 업데이트
                if loss < self.global_best_loss:
                    self.global_best_loss = loss
                    self.global_best = particle.network.get_weights().copy()

            # 진화 (매 5 iteration마다)
            if iteration % 5 == 0:
                self.evolve(X, y)

            # 온도 감소 (탐색 → 수렴)
            self.temperature *= self.temp_decay

            # 추적
            avg_loss = np.mean(losses)
            diversity = self.get_diversity()
            self.history['loss'].append(avg_loss)
            self.history['best_loss'].append(self.global_best_loss)
            self.history['diversity'].append(diversity)
            self.history['temperature'].append(self.temperature)

            if verbose and iteration % 10 == 0:
                print(f"[{iteration:3d}] "
                      f"Avg Loss: {avg_loss:.5f} | "
                      f"Best: {self.global_best_loss:.5f} | "
                      f"Diversity: {diversity:.3f} | "
                      f"Temp: {self.temperature:.3f}")

            # 조기 종료
            if self.global_best_loss < 0.01:
                if verbose:
                    print(f"\n✓ Converged at iteration {iteration}")
                break

        elapsed = time.time() - start

        # 최종: 앙상블 결합 (상위 3개 평균)
        top_particles = sorted(self.particles,
                              key=lambda p: p.best_loss)[:3]
        ensemble_weights = np.mean([p.best_position for p in top_particles], axis=0)

        return {
            'final_loss': self.global_best_loss,
            'iterations': len(self.history['loss']),
            'time': elapsed,
            'best_weights': ensemble_weights
        }


class SGDOptimizer:
    """표준 SGD (비교용)"""

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


def run_qed_experiment(dataset_name='nonlinear'):
    """QED 실험"""
    print(f"\n{'='*80}")
    print(f"🚀 QED vs SGD: {dataset_name.upper()}")
    print(f"{'='*80}\n")

    X, y = make_dataset(dataset_name, n=100)

    # QED
    print("1️⃣  QED (Quantum-Inspired Ensemble Descent)")
    print("-" * 80)
    net_qed = LightweightNN(seed=42)
    opt_qed = QEDOptimizer(
        net_qed,
        n_particles=10,
        learning_rate=0.05,
        temperature_init=0.3,
        temperature_decay=0.97
    )
    result_qed = opt_qed.train(X, y, max_iters=100, verbose=True)

    # SGD
    print(f"\n2️⃣  Standard SGD")
    print("-" * 80)
    net_sgd = LightweightNN(seed=42)
    opt_sgd = SGDOptimizer(net_sgd, learning_rate=0.1)
    result_sgd = opt_sgd.train(X, y, max_iters=100, verbose=True)

    # 비교
    print(f"\n{'='*80}")
    print("📊 결과")
    print(f"{'='*80}")
    print(f"{'지표':<30} {'QED':>25} {'SGD':>20}")
    print("-" * 80)
    print(f"{'최종 손실':<30} {result_qed['final_loss']:>25.6f} {result_sgd['final_loss']:>20.6f}")
    print(f"{'수렴 반복 횟수':<30} {result_qed['iterations']:>25d} {result_sgd['iterations']:>20d}")
    print(f"{'시간 (초)':<30} {result_qed['time']:>25.4f} {result_sgd['time']:>20.4f}")

    improvement = (result_sgd['final_loss'] - result_qed['final_loss']) / result_sgd['final_loss'] * 100

    if improvement > 0:
        print(f"{'🎯 QED 개선율':<30} {improvement:>25.2f}%")
        print("="*80)
        print("✅ QED 승리!")
    else:
        print(f"{'❌ QED 악화':<30} {improvement:>25.2f}%")
        print("="*80)
        print("SGD 승리")

    return {
        'qed': result_qed,
        'sgd': result_sgd,
        'qed_opt': opt_qed,
        'sgd_opt': opt_sgd,
        'dataset': dataset_name
    }


def plot_qed(results):
    """시각화"""
    qed = results['qed_opt']
    sgd = results['sgd_opt']
    ds = results['dataset']

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'QED vs SGD: {ds.upper()}', fontsize=18, fontweight='bold')

    # 1. Loss 비교
    ax = axes[0, 0]
    ax.plot(qed.history['best_loss'], 'b-', label='QED (Best)', linewidth=2.5)
    ax.plot(qed.history['loss'], 'b--', label='QED (Avg)', linewidth=1.5, alpha=0.6)
    ax.plot(sgd.history['loss'], 'r-', label='SGD', linewidth=2.5)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Learning Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 2. Diversity
    ax = axes[0, 1]
    ax.plot(qed.history['diversity'], 'g-', linewidth=2.5)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Diversity (Std)', fontsize=12)
    ax.set_title('Swarm Diversity', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 3. Temperature
    ax = axes[1, 0]
    ax.plot(qed.history['temperature'], 'm-', linewidth=2.5)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Temperature', fontsize=12)
    ax.set_title('Exploration Temperature', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 4. 최종 비교
    ax = axes[1, 1]
    metrics = ['Final Loss', 'Iterations']
    qed_vals = [results['qed']['final_loss'], results['qed']['iterations']]
    sgd_vals = [results['sgd']['final_loss'], results['sgd']['iterations']]

    x = np.arange(len(metrics))
    width = 0.35

    # Normalize for visualization
    qed_norm = [qed_vals[0]*100, qed_vals[1]]
    sgd_norm = [sgd_vals[0]*100, sgd_vals[1]]

    bars1 = ax.bar(x - width/2, qed_norm, width, label='QED', alpha=0.8, color='blue')
    bars2 = ax.bar(x + width/2, sgd_norm, width, label='SGD', alpha=0.8, color='red')

    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filename = f'/Users/say/Documents/GitHub/ai/qed_{ds}_results.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✅ 저장: {filename}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 QED: Quantum-Inspired Ensemble Descent")
    print("   완전히 새로운 최적화 패러다임")
    print("="*80)

    datasets = ['linear', 'nonlinear', 'xor']
    all_results = {}
    wins = 0
    total = len(datasets)

    for ds in datasets:
        result = run_qed_experiment(ds)
        all_results[ds] = result
        plot_qed(result)

        if result['qed']['final_loss'] < result['sgd']['final_loss']:
            wins += 1

    print(f"\n{'='*80}")
    print("🎯 최종 요약")
    print(f"{'='*80}")

    for ds, res in all_results.items():
        qed_loss = res['qed']['final_loss']
        sgd_loss = res['sgd']['final_loss']
        improvement = (sgd_loss - qed_loss) / sgd_loss * 100

        if improvement > 0:
            status = "✅ QED 승리"
        else:
            status = "❌ SGD 승리"

        print(f"{ds.upper():12s} | {status:15s} | {improvement:+7.2f}%")

    print(f"\n{'='*80}")
    print(f"QED 승률: {wins}/{total} ({wins/total*100:.1f}%)")
    print(f"{'='*80}\n")

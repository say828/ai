# ULTIMATE: Step 2 - 상세 설계

## Layer 1: Universal Primitive Pool

### 10개 Universal Primitives 정의

#### P1: GradientDescent
```python
class GradientDescent(Primitive):
    """가장 기본적인 방향: 손실의 기울기"""
    def compute_update(self, network, X, y, lr=0.05):
        grad = network.gradient(X, y)
        return -lr * grad
```

#### P2: MomentumUpdate
```python
class MomentumUpdate(Primitive):
    """관성: 과거 방향 유지"""
    def __init__(self, decay=0.9):
        self.velocity = None

    def compute_update(self, network, X, y, lr=0.05):
        grad = network.gradient(X, y)
        if self.velocity is None:
            self.velocity = np.zeros_like(grad)

        self.velocity = self.decay * self.velocity - lr * grad
        return self.velocity
```

#### P3: AdaptiveStep
```python
class AdaptiveStep(Primitive):
    """적응적 학습률: 성공/실패 기반"""
    def __init__(self):
        self.success_history = deque(maxlen=10)
        self.lr = 0.05

    def compute_update(self, network, X, y):
        grad = network.gradient(X, y)

        # Adapt learning rate
        if len(self.success_history) >= 5:
            recent_success = sum(self.success_history[-5:]) / 5
            if recent_success > 0.7:
                self.lr = min(0.2, self.lr * 1.1)
            elif recent_success < 0.3:
                self.lr = max(0.01, self.lr * 0.9)

        return -self.lr * grad
```

#### P4: ParticleSwarm
```python
class ParticleSwarm(Primitive):
    """집단 탐색: QED의 핵심"""
    def __init__(self, n_particles=5):
        self.particles = [Particle() for _ in range(n_particles)]

    def compute_update(self, network, X, y):
        # 각 particle 업데이트
        for p in self.particles:
            p.update(network, X, y)

        # Best particle의 방향
        best = min(self.particles, key=lambda p: p.loss)
        current = network.get_weights()
        return (best.position - current) * 0.1
```

#### P5: BestAttractor
```python
class BestAttractor(Primitive):
    """최선으로의 끌림: COMP의 BestDirection"""
    def __init__(self):
        self.best_weights = None
        self.best_loss = float('inf')

    def compute_update(self, network, X, y):
        current = network.get_weights()
        loss = network.loss(X, y)

        # Update best
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_weights = current.copy()

        if self.best_weights is None:
            return np.zeros_like(current)

        # Direction to best
        direction = self.best_weights - current
        return direction * 0.1
```

#### P6: StochasticJump
```python
class StochasticJump(Primitive):
    """확률적 점프: 탈출 메커니즘"""
    def compute_update(self, network, X, y, temperature=0.1):
        theta = network.get_weights()
        return np.random.randn(len(theta)) * temperature
```

#### P7: PathSampling
```python
class PathSampling(Primitive):
    """경로 샘플링: PIO의 핵심"""
    def compute_update(self, network, X, y, n_samples=5):
        theta = network.get_weights()
        grad = network.gradient(X, y)

        samples = []
        weights = []

        # Langevin dynamics
        for _ in range(n_samples):
            delta = -0.01 * grad + 0.05 * np.random.randn(len(theta))
            action = 0.5 * np.linalg.norm(delta)**2 + network.loss(X, y)
            weight = np.exp(-action / 0.1)

            samples.append(delta)
            weights.append(weight)

        # Weighted average
        weights = np.array(weights) / sum(weights)
        return sum(w * s for w, s in zip(weights, samples))
```

#### P8: ActionGuided
```python
class ActionGuided(Primitive):
    """Action 기반: LAML-Q의 핵심"""
    def compute_update(self, network, X, y):
        theta = network.get_weights()
        grad = network.gradient(X, y)

        # Candidate updates
        candidates = [
            -0.01 * grad,  # Small step
            -0.05 * grad,  # Medium step
            -0.1 * grad,   # Large step
        ]

        # Compute action for each
        actions = []
        for delta in candidates:
            kinetic = 0.5 * np.linalg.norm(delta)**2
            network.set_weights(theta + delta)
            potential = network.loss(X, y)
            action = kinetic + potential
            actions.append(action)

        # Reset
        network.set_weights(theta)

        # Choose minimum action
        best_idx = np.argmin(actions)
        return candidates[best_idx]
```

#### P9: MultiScale
```python
class MultiScale(Primitive):
    """다중 시간 척도: LAML-Q의 핵심"""
    def compute_update(self, network, X, y):
        theta = network.get_weights()
        grad = network.gradient(X, y)

        # Different scales
        updates = [
            -0.01 * grad,  # 1-step
            -0.05 * grad,  # 5-step
            -0.1 * grad,   # 10-step
        ]

        # Quality of each scale
        qualities = []
        for delta in updates:
            network.set_weights(theta + delta)
            loss = network.loss(X, y)
            quality = 1.0 / (loss + 1e-8)
            qualities.append(quality)

        # Reset
        network.set_weights(theta)

        # Weighted average
        qualities = np.array(qualities) / sum(qualities)
        return sum(q * u for q, u in zip(qualities, updates))
```

#### P10: EnsembleAverage
```python
class EnsembleAverage(Primitive):
    """앙상블: 여러 방법의 조합"""
    def compute_update(self, network, X, y):
        theta = network.get_weights()
        grad = network.gradient(X, y)

        # Different strategies
        strategies = [
            -0.05 * grad,  # Gradient
            -0.05 * grad + 0.01 * np.random.randn(len(grad)),  # + noise
            -0.05 * grad * np.random.choice([0.5, 1.0, 2.0], len(grad)),  # Variable LR
        ]

        # Simple average
        return sum(strategies) / len(strategies)
```

---

## Layer 2: Strategy Selector

### Context Computation

```python
def compute_context(network, X, y, history):
    """
    상황을 벡터로 표현

    Returns:
        np.array of shape (context_dim,)
    """
    theta = network.get_weights()
    grad = network.gradient(X, y)
    loss = network.loss(X, y)

    context = []

    # 1. 현재 상태
    context.append(loss)  # 현재 손실
    context.append(np.linalg.norm(grad))  # 기울기 크기
    context.append(np.mean(np.abs(theta)))  # 가중치 평균 크기

    # 2. 안정성
    if len(history['losses']) > 5:
        recent_losses = history['losses'][-5:]
        context.append(np.var(recent_losses))  # 손실 분산
        context.append(np.mean(np.diff(recent_losses)))  # 손실 변화율
    else:
        context.extend([1.0, 0.0])

    # 3. 진행 상황
    context.append(history['iteration'] / history['max_iters'])  # 진행도
    if len(history['losses']) > 1:
        improvement = history['losses'][-2] - history['losses'][-1]
        context.append(improvement / history['losses'][-2])  # 상대 개선
    else:
        context.append(0.0)

    # 4. 성공률
    if len(history['successes']) > 0:
        context.append(np.mean(history['successes'][-10:]))  # 최근 성공률
    else:
        context.append(0.5)

    # 5. 문제 특성 (간단한 휴리스틱)
    # Hessian diagonal 근사로 smoothness 추정
    grad2 = compute_hessian_diagonal_approx(network, X, y)
    context.append(np.mean(np.abs(grad2)))  # Curvature

    # 6. 차원
    context.append(np.log(len(theta)))  # log(dimensionality)

    return np.array(context, dtype=np.float32)
```

### Policy Network Architecture

```python
class PolicyNetwork(nn.Module):
    """
    Context → Primitive Weights

    Input: context vector (11-dim)
    Output: primitive weights (10-dim, summing to 1)
    """
    def __init__(self, context_dim=11, n_primitives=10):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),

            nn.Linear(64, n_primitives),
            nn.Softmax(dim=-1)
        )

        # Initialize to uniform
        with torch.no_grad():
            self.encoder[-2].weight.fill_(0.0)
            self.encoder[-2].bias.fill_(0.0)

    def forward(self, context):
        """
        Args:
            context: Tensor of shape (batch, context_dim) or (context_dim,)

        Returns:
            weights: Tensor of shape (batch, n_primitives) or (n_primitives,)
        """
        if len(context.shape) == 1:
            context = context.unsqueeze(0)

        weights = self.encoder(context)

        if weights.shape[0] == 1:
            weights = weights.squeeze(0)

        return weights
```

---

## Layer 3: Meta-Learner

### Experience Collection

```python
class ExperienceBuffer:
    """경험 저장 및 관리"""
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def add(self, context, weights, improvement, problem_id=None):
        """
        경험 추가

        Args:
            context: 상황 벡터
            weights: 사용한 primitive weights
            improvement: 결과 (loss 감소량)
            problem_id: 어떤 문제인지 (선택)
        """
        self.buffer.append({
            'context': context,
            'weights': weights,
            'improvement': improvement,
            'problem_id': problem_id,
            'timestamp': time.time()
        })

    def sample(self, batch_size=64):
        """무작위 샘플링"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        contexts = np.stack([e['context'] for e in batch])
        weights = np.stack([e['weights'] for e in batch])
        improvements = np.array([e['improvement'] for e in batch])

        return contexts, weights, improvements
```

### Meta-Training

```python
class MetaLearner:
    """
    Policy network를 경험으로부터 학습

    Objective:
        Policy가 높은 improvement를 가져오는 weights를 예측하도록
    """
    def __init__(self, policy_net):
        self.policy_net = policy_net
        self.optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4)

    def update(self, experience_buffer, n_epochs=10, batch_size=64):
        """
        Experience buffer로부터 학습

        Loss: Weighted MSE
          - Policy가 예측한 weights
          - 실제로 효과적이었던 weights
          - Improvement로 가중치
        """
        if len(experience_buffer.buffer) < batch_size:
            return

        for epoch in range(n_epochs):
            # Sample batch
            contexts, target_weights, improvements = experience_buffer.sample(batch_size)

            # To tensors
            contexts = torch.FloatTensor(contexts)
            target_weights = torch.FloatTensor(target_weights)
            improvements = torch.FloatTensor(improvements)

            # Normalize improvements (weights for loss)
            improvements = improvements - improvements.min()
            improvements = improvements / (improvements.max() + 1e-8)

            # Forward
            predicted_weights = self.policy_net(contexts)

            # Loss: Weighted MSE
            # 더 많이 개선된 경험에 더 큰 가중치
            mse = (predicted_weights - target_weights)**2
            weighted_mse = (mse * improvements.unsqueeze(1)).mean()

            # KL divergence regularization (prevent overconfident)
            uniform = torch.ones_like(predicted_weights) / predicted_weights.shape[1]
            kl_div = (predicted_weights * (torch.log(predicted_weights + 1e-8) - torch.log(uniform))).sum(1).mean()

            loss = weighted_mse + 0.01 * kl_div

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()

    def evaluate(self, test_problems):
        """
        테스트 문제들에서 성능 평가
        """
        results = []
        for problem in test_problems:
            # Run optimization with learned policy
            result = run_optimization_with_policy(problem, self.policy_net)
            results.append(result)

        return {
            'mean_final_loss': np.mean([r['final_loss'] for r in results]),
            'mean_iterations': np.mean([r['iterations'] for r in results]),
            'success_rate': np.mean([r['converged'] for r in results]),
        }
```

---

## 전체 ULTIMATE Optimizer

### Main Class

```python
class ULTIMATE_Optimizer:
    """
    Meta-Conscious Optimizer

    3-Layer Architecture:
      1. Primitive Pool: 10 universal primitives
      2. Strategy Selector: Context → Weights (learned)
      3. Meta-Learner: Experience → Policy improvement
    """
    def __init__(self, network, pretrained_policy=None):
        """
        Args:
            network: Neural network to optimize
            pretrained_policy: Pre-trained policy network (optional)
        """
        # Layer 1: Primitives
        self.primitives = [
            GradientDescent(),
            MomentumUpdate(),
            AdaptiveStep(),
            ParticleSwarm(n_particles=3),
            BestAttractor(),
            StochasticJump(),
            PathSampling(n_samples=3),
            ActionGuided(),
            MultiScale(),
            EnsembleAverage(),
        ]

        # Layer 2: Policy
        if pretrained_policy is not None:
            self.policy_net = pretrained_policy
        else:
            self.policy_net = PolicyNetwork()

        # Layer 3: Meta-learner
        self.meta_learner = MetaLearner(self.policy_net)
        self.experience_buffer = ExperienceBuffer()

        # Tracking
        self.network = network
        self.history = {
            'iteration': 0,
            'max_iters': 100,
            'losses': [],
            'successes': [],
            'contexts': [],
            'weights_used': [],
        }

    def step(self, X, y):
        """Single optimization step"""
        # 1. Compute context
        context = compute_context(self.network, X, y, self.history)

        # 2. Get strategy from policy
        context_tensor = torch.FloatTensor(context)
        with torch.no_grad():
            weights = self.policy_net(context_tensor).numpy()

        # 3. Execute primitives
        updates = []
        for primitive in self.primitives:
            update = primitive.compute_update(self.network, X, y)
            updates.append(update)

        # 4. Combine
        final_update = sum(w * u for w, u in zip(weights, updates))

        # 5. Apply
        old_loss = self.network.loss(X, y)
        theta = self.network.get_weights()
        self.network.set_weights(theta + final_update)
        new_loss = self.network.loss(X, y)

        # 6. Record
        improvement = old_loss - new_loss
        success = improvement > 0

        self.history['losses'].append(new_loss)
        self.history['successes'].append(1 if success else 0)
        self.history['contexts'].append(context)
        self.history['weights_used'].append(weights)
        self.history['iteration'] += 1

        # 7. Add to experience buffer
        self.experience_buffer.add(context, weights, improvement)

        # 8. Meta-learning (periodic)
        if len(self.experience_buffer.buffer) >= 100 and \
           self.history['iteration'] % 50 == 0:
            self.meta_learner.update(self.experience_buffer, n_epochs=5)

        return new_loss, weights

    def train(self, X, y, max_iters=100, tolerance=1e-3, verbose=True):
        """Full training loop"""
        self.history['max_iters'] = max_iters

        for i in range(max_iters):
            loss, weights = self.step(X, y)

            if verbose and i % 10 == 0:
                dominant_idx = np.argmax(weights)
                dominant_name = self.primitives[dominant_idx].__class__.__name__
                print(f"[{i:3d}] Loss: {loss:.5f} | Dominant: {dominant_name}")

            # Convergence check
            if i > 10 and abs(self.history['losses'][-1] - self.history['losses'][-2]) < tolerance:
                if verbose:
                    print(f"\n✓ Converged at iteration {i}")
                break

        return {
            'final_loss': self.history['losses'][-1],
            'losses': self.history['losses'],
            'iterations': len(self.history['losses']),
            'weights_history': self.history['weights_used'],
        }
```

---

## Step 2 결론

### 완성된 설계

**Layer 1**: 10개 Universal Primitives ✅
- 모든 성공 방법의 정수
- 각각 독립적으로 작동
- 조합 가능

**Layer 2**: Policy Network ✅
- Context (11-dim) → Weights (10-dim)
- Neural network
- 학습 가능

**Layer 3**: Meta-Learner ✅
- Experience buffer
- Weighted MSE loss
- Online learning

### 예상 효과

1. **Adaptation**: 문제 자동 인식, 전략 자동 선택
2. **Learning**: 경험 축적, 성능 개선
3. **Combination**: 최선의 조합 자동 발견
4. **Transparency**: Weights로 설명 가능
5. **Evolution**: 사용할수록 똑똑해짐

### 다음 스텝

Step 3에서:
1. 구현 (코드 작성)
2. 실험 (3개 데이터셋)
3. 결과 분석

---

**작성**: 2026-01-03 ULTIMATE Step 2/4
**다음**: Step 3 - 구현 및 실험

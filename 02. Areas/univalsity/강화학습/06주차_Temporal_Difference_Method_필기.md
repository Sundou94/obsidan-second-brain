---
title: "Temporal Difference Method"
subtitle: "TD 법, SARSA, Q Learning"
subject: "강화학습"
week: 6
professor: "박태형 (충북대 지능시스템로봇공학과)"
date_created: 2026-04-08
tags:
  - 강화학습
  - TD법
  - SARSA
  - Q-Learning
  - 정책평가
  - On-Policy
  - Off-Policy
  - Importance-Sampling
status: 학습중
---

# Temporal Difference Method

## 목차

- [[#1. TD 법 개요 MC vs TD]]
- [[#2. Policy Evaluation (정책 평가)]]
  - [[#2-1. DP법]]
  - [[#2-2. MC법]]
  - [[#2-3. TD법]]
  - [[#2-4. TD 법 구현 코드]]
- [[#3. SARSA]]
  - [[#3-1. SARSA 개념]]
  - [[#3-2. On-Policy vs Off-Policy]]
  - [[#3-3. On-Policy SARSA 구현]]
  - [[#3-4. Off-Policy SARSA]]
  - [[#3-5. Importance Sampling]]
- [[#4. Q Learning]]
  - [[#4-1. SARSA vs Q Learning 비교]]
  - [[#4-2. Q Learning이 Off-Policy인 이유]]
  - [[#4-3. Q Learning 구현]]
  - [[#4-4. Q Learning 역사]]
- [[#🧠 내 생각 / 의문점]]
- [[#📝 시험 대비 핵심 정리]]
- [[#❓ 퀴즈]]

---

## 1. TD 법 개요: MC vs TD

### MC (Monte Carlo) 법

- 에피소드가 **끝**에 도달한 후 가치함수(Q) 계산 → 정책평가(policy evaluation) → 정책갱신(policy update)
- **일회성 과제 (O), 지속성 과제 (X)**
  - 에피소드가 반드시 끝나야 학습 가능 → 끝이 없는 환경에는 적용 불가

> [!tip] 용어 설명: 에피소드 (Episode)
> 에이전트가 시작 상태에서 종료 상태까지 행동하는 하나의 "게임 한 판"을 의미한다. 예를 들어 바둑을 두는 것이라면 한 판의 바둑이 하나의 에피소드다.

> [!tip] 용어 설명: 일회성 과제 vs 지속성 과제
> - **일회성 과제 (Episodic Task)**: 명확한 종료 시점이 있는 과제 (예: 게임의 승패)
> - **지속성 과제 (Continuing Task)**: 종료 없이 계속되는 과제 (예: 로봇이 계속 걸어다니는 것)

### TD (Temporal Difference, 시간차) 법

- 에피소드가 끝날 때까지 **기다리지 않고**, 일정 시간마다 정책평가 및 정책갱신
- **일회성 과제 (O), 지속성 과제 (O)** → 두 종류 모두 적용 가능
- **TD 법 = MC 법 + DP 법**의 장점 결합

> [!info] 보충 설명: TD = MC + DP
> - MC 법의 장점: 환경 모델(전이 확률) 없이 샘플링만으로 학습 가능
> - DP 법의 장점: 에피소드 끝까지 기다리지 않고 bootstrap(추정값으로 추정)으로 순차적 갱신 가능
> - TD 법은 이 두 장점을 모두 취한다.

| 방법 | 환경 모델 필요? | 에피소드 끝 필요? | 일회성 | 지속성 |
|------|:-----------:|:-----------:|:----:|:----:|
| DP 법 | O | X | O | O |
| MC 법 | X | O | O | X |
| TD 법 | X | X | O | O |

---

## 2. Policy Evaluation (정책 평가)

### 핵심 개념: Return과 Value Function

**Return (수익)**:
$$G_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \cdots = R_t + \gamma G_{t+1}$$

**Value Function (가치 함수)**:
$$v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s] = \mathbb{E}_\pi[R_t + \gamma G_{t+1} \mid S_t = s]$$

> [!tip] 용어 설명: γ (감가율, Discount Factor)
> 미래 보상을 현재 가치로 환산하는 비율. 0 < γ < 1 이며, γ가 작을수록 먼 미래보상을 덜 중요시한다. 예: 오늘 받는 1만원이 1년 후 받는 1만원보다 가치 있는 것과 같은 개념.

---

### 2-1. DP법

- 다음 상태의 가치로부터 현재 상태의 가치를 추정 (**Bootstrap**)
- 환경에 대한 모델링을 통해 **모든 경우**의 상태 전이를 고려

$$V_\pi'(s) = \sum_{a,s'} \pi(a|s)p(s'|s,a)\{r(s,a,s') + \gamma V_\pi(s')\} \quad \text{(Bellman equation)}$$

> [!tip] 용어 설명: Bootstrap (부트스트랩)
> 아직 완전히 계산되지 않은 추정값을 다른 추정값 계산에 재사용하는 방법. "자기 자신의 추정으로 자기 자신을 업데이트"하는 방식이다. DP와 TD 법에서 사용된다.

---

### 2-2. MC법

- **특정한 sample data** (현재상태 → 최종상태)의 return 평균을 사용
- 지수 이동 평균(Exponential Moving Average) 방식으로 가치함수 업데이트

$$V_\pi'(S_t) = V_\pi(S_t) + \alpha\{G_t - V_\pi(S_t)\}$$

- $G_t$: 목표에 도달 시 얻을 수 있는 수익 → **현재~목표까지의 샘플 필요**

---

### 2-3. TD법

- DP처럼 bootstrap을 통해 가치함수를 순차적으로 갱신 (다음상태 → 현재상태)
- MC처럼 환경에 대한 정보 없이 **sampling된 데이터만으로** 가치함수 갱신

$$\boxed{V_\pi'(S_t) = V_\pi(S_t) + \alpha\{R_t + \gamma V_\pi(S_{t+1}) - V_\pi(S_t)\}}$$

**각 항의 의미:**

| 항 | 의미 |
|----|------|
| $V_\pi(S_t)$ | current-state $S_t$에서 목표에 도달 시 얻을 수 있는 수익 **(추정치)** |
| $V_\pi(S_{t+1})$ | next-state $S_{t+1}$에서 목표에 도달 시 얻을 수 있는 수익 **(추정치)** |
| $R_t + \gamma V_\pi(S_{t+1})$ | **TD Target**: 목표에 도달 시 얻을 수 있는 수익 (추정치), 현재→다음까지의 샘플만 필요 |

**MC vs TD 비교:**

```
〈MC법〉  V'(St) = Vπ(St) + α { Gt - Vπ(St) }
                                ↑
                        현재~목표까지 전체 샘플 필요

〈TD법〉  V'(St) = Vπ(St) + α { Rt + γVπ(St+1) - Vπ(St) }
                                ↑
                        현재→다음 한 스텝의 샘플만 필요
```

> [!info] 보충 설명: TD법이 MC법 대비 데이터 변동성 감소
> MC법은 에피소드 전체를 거쳐야 $G_t$를 계산하므로, 경로에 따른 변동성(variance)이 크다. TD법은 한 스텝 앞만 보므로 변동성이 훨씬 작고 안정적이다.

---

### 2-4. TD 법 구현 코드

**TdAgent Class (핵심 구조):**

```python
class TdAgent:
    def __init__(self):
        self.gamma = 0.9      # 감가율
        self.alpha = 0.01     # 학습률
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)  # 정책
        self.V = defaultdict(lambda: 0)                # 가치함수

    def eval(self, state, reward, next_state, done):
        next_V = 0 if done else self.V[next_state]  # 목표 지점의 가치함수는 0
        target = reward + self.gamma * next_V
        self.V[state] += (target - self.V[state]) * self.alpha
        # → V'(St) = Vπ(St) + α{Rt + γVπ(St+1) - Vπ(St)}
```

> [!warning] 주의: MC법과 TD법의 eval 호출 시점 차이
> - **TD법**: `agent.eval()` → **매 step마다** 호출 (while True 루프 안)
> - **MC법**: `agent.eval()` → **목표 도달 시(if done)** 호출
>
> 이 차이가 TD법의 핵심이다! TD는 에피소드가 끝나지 않아도 매 스텝 학습한다.

**실행 루프 (TD법):**

```python
env = GridWorld()
agent = TdAgent()

for episode in range(1000):
    state = env.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.eval(state, reward, next_state, done)  # ← 매번 호출!
        if done:
            break
        state = next_state

env.render_v(agent.V)
```

---

## 3. SARSA

### 3-1. SARSA 개념

SARSA는 **TD법으로 Q함수(Action-value function)를 구하고, ε-greedy로 정책을 개선하는 방법**이다.

**Value Function Update (TD법):**

- State-value function:
$$V_\pi'(S_t) = V_\pi(S_t) + \alpha\{R_t + \gamma V_\pi(S_{t+1}) - V_\pi(S_t)\}$$

- **Action-value function (Q함수):**
$$\boxed{Q_\pi'(S_t, A_t) = Q_\pi(S_t, A_t) + \alpha\{R_t + \gamma Q_\pi(S_{t+1}, A_{t+1}) - Q_\pi(S_t, A_t)\}}$$

**Policy Update (ε-greedy):**
$$\pi'(a|S_t) = \begin{cases} \text{argmax}_a Q_\pi(S_t, a) & (1-\varepsilon \text{의 확률}) \\ \text{무작위 행동} & (\varepsilon \text{의 확률}) \end{cases}$$

> [!tip] 용어 설명: Q함수 (Action-value function)
> 상태(State)뿐만 아니라 **행동(Action)**까지 고려한 가치함수. Q(s, a)는 "상태 s에서 행동 a를 취했을 때 얻을 수 있는 기대 수익"을 의미한다. V(s)가 상태의 가치라면, Q(s,a)는 (상태, 행동) 쌍의 가치다.

> [!tip] 용어 설명: ε-greedy 정책
> 대부분의 경우(1-ε 확률) Q값이 가장 높은 행동을 선택(탐욕적 선택, exploitation)하고, 일부 경우(ε 확률) 무작위 행동을 선택(탐험, exploration)하는 정책. 탐험과 활용의 균형을 맞추는 방법이다.

**SARSA 이름의 유래:**

```
(St, At, Rt, St+1, At+1)
  S   A   R   S    A
```

한 번의 업데이트에 필요한 5개의 요소 **(State, Action, Reward, State, Action)**의 앞 글자를 따서 SARSA라고 한다.

---

**SARSA 구현 핵심 코드:**

```python
class SarsaAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        self.memory = deque(maxlen=2)  # FIFO, 최근 2개 데이터만 보관

    def update(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))
        if len(self.memory) < 2:
            return

        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]  # ❸ 다음 Q함수

        next_q = 0 if done else self.Q[next_state, next_action]

        # ❹ TD법으로 self.Q 갱신
        target = reward + self.gamma * next_q
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        # ❺ 정책 개선
        self.pi[state] = greedy_probs(self.Q, state, self.epsilon)
```

> [!info] 보충 설명: deque(maxlen=2) 사용 이유
> SARSA는 (St, At, Rt, St+1, At+1)을 한 묶음으로 업데이트한다. 즉, 현재 상태와 **다음 행동(At+1)**까지 알아야 Q값을 업데이트할 수 있다. 따라서 최근 2개의 (state, action, reward, done) 데이터를 버퍼에 보관해야 한다.

**실행 시 주의사항:**

```python
while True:
    action = agent.get_action(state)
    next_state, reward, done = env.step(action)

    agent.update(state, action, reward, done)  # ❶ 매번 호출
    if done:
        # ❷ 목표에 도달했을 때도 호출 (update() 는 2번 호출을 1세트로 정책갱신)
        agent.update(next_state, None, None, None)
        break
    state = next_state
```

---

### 3-2. On-Policy vs Off-Policy

| 구분 | On-Policy | Off-Policy |
|------|-----------|------------|
| **정의** | 스스로 쌓은 경험으로 자신의 정책을 개선 | 자신과 다른 환경(다른 정책)에서 쌓은 경험으로 자신의 정책을 개선 |
| **대상 정책(target policy)과 행동 정책(behavior policy)** | 동일함 | 구분됨 |
| **예시** | 직접 운전 연습으로 실력 향상 | 다른 테니스 선수가 스윙하는 모습을 보고 자신의 스윙 자세를 고침 |
| **특성** | 안정적이나 비효율적 | 최적성 향상, 중요도 샘플링 필요 |
| **대표 알고리즘** | SARSA | Q Learning, Off-Policy SARSA |

> [!tip] 용어 설명: 대상 정책(Target Policy)과 행동 정책(Behavior Policy)
> - **대상 정책(π)**: 학습/개선하고자 하는 정책
> - **행동 정책(b)**: 실제로 환경과 상호작용하며 데이터를 수집하는 정책
> - On-Policy에서는 π = b, Off-Policy에서는 π ≠ b

> [!info] 보충 설명: Off-Policy의 장점
> Off-Policy는 이미 수집된 데이터를 다른 정책 학습에도 재활용할 수 있다. 또한 행동 정책(b)은 탐험(exploration)에 집중하고, 대상 정책(π)은 탐욕적(greedy)으로 최적화할 수 있어 정책의 최적성이 향상된다.

---

### 3-3. On-Policy SARSA 구현

- 위에서 설명한 SARSA가 바로 On-Policy SARSA다.
- `action_probs = self.pi[state]` → **자신의 정책(π)**에서 행동을 선택
- 실습 파일: `sarsa.py` (실습 #2)
- 에피소드 수: 10,000번

---

### 3-4. Off-Policy SARSA

**행동 정책과 대상 정책의 분리:**

- **대상 정책(π)**: Policy upgrade → **greedy** (exploitation, 탐욕적 선택)
- **행동 정책(b)**: Policy upgrade → **ε-greedy** (exploration, 탐험 포함)

**행동을 선택할 때:** `action_probs = self.b[state]` → **행동 정책(b)**에서 행동 선택

**중요도 비율(ρ):**
$$\rho = \frac{\pi(A_{t+1}|S_{t+1})}{b(A_{t+1}|S_{t+1})}$$

**TD Target 보정:**
$$Q_\pi'(S_t, A_t) = Q_\pi(S_t, A_t) + \alpha\{\rho(R_t + \gamma Q_\pi(S_{t+1}, A_{t+1})) - Q_\pi(S_t, A_t)\}$$

> [!warning] 주의: On-Policy SARSA와 Off-Policy SARSA의 TD Target 차이
> - **On-Policy**: TD target = $R_t + \gamma Q(S_{t+1}, A_{t+1})$ (보정 없음)
> - **Off-Policy**: TD target = $\rho(R_t + \gamma Q(S_{t+1}, A_{t+1}))$ (ρ로 보정)
>
> 행동 정책과 대상 정책이 다르기 때문에, 중요도 비율 ρ를 곱해 TD Target을 보정해야 한다.



**구현 핵심:**

```python
# ❷ 가중치 rho 계산
rho = self.pi[next_state][next_action] / self.b[next_state][next_action]

# ❸ rho로 TD 목표 보정
target = rho * (reward + self.gamma * next_q)
self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

# ❹ 각각의 정책 개선
self.pi[state] = greedy_probs(self.Q, state, 0)      # greedy (대상 정책)
self.b[state] = greedy_probs(self.Q, state, self.epsilon)  # ε-greedy (행동 정책)
```

---

### 3-5. Importance Sampling (중요도 샘플링)

**정의:**
확률분포 π의 기대값을 다른 확률분포(b)에서 샘플링한 데이터를 사용하여 계산하는 기법

**수학적 원리:**
$$\mathbb{E}_\pi[x] = \sum_x x\pi(x) = \sum_x x \frac{b(x)}{b(x)}\pi(x) = \sum_x x \frac{\pi(x)}{b(x)}b(x) = \mathbb{E}_b\left[x\frac{\pi(x)}{b(x)}\right]$$

**방법:**
1. 확률분포 b에서 샘플링: $x^{(i)} \sim b$ $(i = 1, \cdots, n)$
2. 중요도 비율 계산: $\rho(x) = \frac{\pi(x)}{b(x)}$
3. 가중 평균 계산:

$$\mathbb{E}_\pi[x] \approx \frac{\rho(x^{(1)})x^{(1)} + \cdots + \rho(x^{(n)})x^{(n)}}{n}$$

> [!tip] 용어 설명: Importance Sampling (중요도 샘플링)
> 우리가 원하는 분포(π)에서 직접 샘플링하기 어려울 때, 다른 분포(b)에서 샘플링한 후 가중치(ρ = π/b)를 곱해 보정하는 방법이다. Off-Policy 강화학습에서 행동 정책(b)으로 모은 데이터로 대상 정책(π)의 기댓값을 추정할 때 사용한다.

> [!warning] 주의: 두 분포가 유사할수록 샘플링의 분산이 작아짐
> - π와 b가 비슷할수록 ρ ≈ 1 → 분산(variance) 작음 → 안정적
> - π와 b가 다를수록 ρ 값이 크게 달라짐 → 분산 커짐 → 불안정
>
> 실험 결과: b = [1/3, 1/3, 1/3] (균등분포)일 때
> - 몬테카를로법: 2.78 (분산: 0.27)
> - 중요도 샘플링: 2.95 (분산: 10.63)
>
> b가 π와 달라 분산이 매우 커졌다! → Off-Policy SARSA의 단점

---

## 4. Q Learning


이거 시험문제에 나올듯 
SARSA 와 Qlearning 의 차이를 설명하여라 

### 4-1. SARSA vs Q Learning 비교

| 구분 | SARSA | Q Learning |
|------|-------|-----------|
| **Bellman 방정식** | Bellman equation | **Bellman optimality equation** |
| **다음 행동 선택** | π(a\|s)에서 샘플링 | **Q함수가 가장 큰(MAX) 행동으로 선택** |
| **중요도 샘플링** | 필요 (Off-Policy일 때) | **불필요** |
| **Q 갱신 안정성** | 불안정 (Off-Policy 시) | **안정** |
| **최적화 성능** | 보통 | **향상** |

**SARSA의 Q 갱신 공식 (On-Policy):**
$$Q'(S_t, A_t) = Q_\pi(S_t, A_t) + \alpha\{R_t + \gamma Q_\pi(S_{t+1}, A_{t+1}) - Q_\pi(S_t, A_t)\}$$

**Q Learning의 Q 갱신 공식:**
$$\boxed{Q'(S_t, A_t) = Q(S_t, A_t) + \alpha\{R_t + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)\}}$$

> [!tip] 용어 설명: Bellman Optimality Equation (벨만 최적 방정식)
> 일반 벨만 방정식이 특정 정책의 가치를 계산하는 반면, 벨만 최적 방정식은 **최적 정책**의 가치를 계산한다. 핵심은 다음 상태에서 가능한 모든 행동 중 **최대(MAX) Q값**을 사용한다는 점이다.

> [!info] 보충 설명: Q Learning에서 π가 불필요한 이유
> SARSA에서 Off-Policy를 구현하려면 대상 정책(π)이 별도로 필요하다. 하지만 Q Learning은 다음 행동을 "Q값이 최대인 행동"으로 자동 결정하므로, 별도의 대상 정책(π)을 관리할 필요가 없다. 행동 정책(b)으로 모은 데이터에서 max Q값만 취하면 된다.

---

### 4-2. Q Learning이 Off-Policy인 이유

**SARSA (Off-Policy)의 문제점:**
- $A_{t+1}$을 확률분포 π 또는 b에서 샘플링
- 중요도 샘플링 필요 → 샘플링의 분산이 커짐 → **Q 갱신 불안정**

**Q Learning의 해결책:**
- $A_{t+1}$을 **Q함수가 가장 큰(MAX) 행동으로 선택** (π 불필요)
- 중요도 샘플링 불필요 → **Q 갱신 안정**
- 최적화 성능 향상

```
〈SARSA〉
  갱신 대상 → π(a|s) 또는 b(a|s) 중 하나를 선택해야 함
  →  중요도 샘플링(ρ) 보정 필요

〈Q Learning〉
  다음 행동 = argmax Q(St+1, a)  ← 항상 최대값 선택
  → 별도 π 없이도 Off-Policy 학습 가능!
```

---

### 4-3. Q Learning 구현

```python
class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        self.b = defaultdict(lambda: random_actions)  # 행동 정책 (ε-greedy)
        self.Q = defaultdict(lambda: 0)

    def get_action(self, state):
        action_probs = self.b[state]  # 행동 정책에서 가져옴
        ...

    def update(self, state, action, reward, next_state, done):
        if done:
            next_q_max = 0
        else:
            # 다음 상태에서 Q함수의 최댓값 계산
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max = max(next_qs)

        # Q함수 갱신
        target = reward + self.gamma * next_q_max
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        # 행동 정책 갱신 (ε-greedy)
        self.b[state] = greedy_probs(self.Q, state, self.epsilon)
```

> [!info] 보충 설명: SARSA와 Q Learning 구현의 핵심 차이
> - **SARSA**: `next_q = self.Q[next_state, next_action]` → 다음 행동까지 알아야 함 (deque 필요)
> - **Q Learning**: `next_q_max = max(self.Q[next_state, a] for a in ...)` → 다음 행동 없이 max로 결정
>
> 따라서 Q Learning에서는 SARSA의 deque가 필요 없다! update() 함수 시그니처도 다르다.

**실행 루프:**

```python
for episode in range(10000):
    state = env.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.update(state, action, reward, next_state, done)  # 매번 호출
        if done:
            break
        state = next_state
```

---

### 4-4. Q Learning 역사

```
1989  Q-Learning 등장
      - Watkins & Dayan, "Q-Learning", Machine Learning
      - Q-table 학습

1990s~2000s 초반  알고리즘 확장
      - SARSA (On-policy), Dyna-Q 등
      - 게임, 로보틱스 등에서 실험적으로 사용

2013  Deep Q-Network (DQN) 등장 ★★★
      - DeepMind (Mnih et al.), "Playing Atari with Deep Reinforcement Learning"
      - Q-table 대신 신경망으로 Q-value를 근사
      - 인간 수준의 Atari 게임 플레이 달성

최근  다양한 DQN 변형 등장
      - Double DQN, Dueling DQN, Rainbow DQN ...
```

> [!info] 보충 설명: Q Learning의 의의
> Q Learning은 강화학습의 핵심 기법으로, Q함수를 효율적이고 안정적으로 갱신하며, 정책 결정의 최적화 성능을 향상시킨다. 2013년 DQN의 등장으로 딥러닝과 결합하면서 강화학습의 실용성이 폭발적으로 향상되었고, 현재 많은 연구의 기반 알고리즘으로 사용된다.

> [!tip] 용어 설명: DQN (Deep Q-Network)
> Q-table(표) 대신 딥 신경망(Deep Neural Network)을 사용해 Q함수를 근사하는 방법. Q-table은 상태와 행동의 수가 적을 때만 사용 가능하지만, DQN은 픽셀 같은 고차원 상태도 처리 가능하다.

**Q Learning 요약:**
- 상태(state)와 행동(action)에 대해 Q-value를 업데이트하면서, 보상을 최대화하는 정책(policy)을 학습
- TD법으로 가치함수 평가 (MC와 같이 샘플링 데이터 기반, '지금'과 '다음' 정보만 사용)
- Bellman optimality equation 사용
- **Off-Policy이지만 중요도 샘플링 사용하지 않음** → 안정적 갱신

---

## 🧠 내 생각 / 의문점

> *이 공간에 수업을 듣고 떠오르는 생각이나 의문점을 자유롭게 작성해 보세요*

### 💡 인사이트

-

### ❓ 의문점

- Q Learning이 항상 SARSA보다 좋다면, 왜 아직도 SARSA를 사용하는 걸까? 어떤 상황에서 SARSA가 더 유리할까?
- ε의 값은 어떻게 결정하는가? 학습 초반에는 크게 하고 후반에는 줄이는 방식(ε-decay)이 더 좋지 않을까?
- DQN에서 Q-table을 신경망으로 대체하면 구체적으로 어떻게 학습이 이루어지는가?

### 🔗 연관 개념

- 이전 주차 내용과 연결되는 부분: MC법(5주차)과 DP법의 한계를 극복하는 TD법, 5주차의 ε-greedy가 SARSA/Q-Learning에도 그대로 적용됨
- 실제 적용 가능한 프로젝트 아이디어: GridWorld 외에 미로 탈출, 간단한 게임에 Q Learning 적용 실험

---

## 📝 시험 대비 핵심 정리

> **★★★ 반드시 암기할 내용**

### 1. DP / MC / TD 비교

| | DP | MC | TD |
|--|:--:|:--:|:--:|
| 환경 모델 필요 | O | X | X |
| 에피소드 완료 필요 | X | O | X |
| 일회성 과제 | O | O | O |
| 지속성 과제 | O | X | O |
| 업데이트 방식 | Bootstrap (모든 경우) | 샘플 return 평균 | Bootstrap + 샘플 |

### 2. TD 법 핵심 공식

$$V'(S_t) = V(S_t) + \alpha\{R_t + \gamma V(S_{t+1}) - V(S_t)\}$$

- **TD Target**: $R_t + \gamma V(S_{t+1})$
- MC와의 차이: MC는 $G_t$(에피소드 끝까지의 수익), TD는 $R_t + \gamma V(S_{t+1})$(한 스텝 앞)

### 3. SARSA 핵심 공식

$$Q'(S_t, A_t) = Q(S_t, A_t) + \alpha\{R_t + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)\}$$

- **(S, A, R, S, A)** 5개 요소 필요
- On-Policy: 행동 정책 = 대상 정책 (ε-greedy)
- deque(maxlen=2)로 2개의 (s, a, r, done) 데이터 보관

### 4. On-Policy vs Off-Policy

| | On-Policy | Off-Policy |
|--|-----------|------------|
| 대표 알고리즘 | SARSA | Q Learning, Off-Policy SARSA |
| target policy = behavior policy | O | X |
| 중요도 샘플링 | 불필요 | 필요 (단, Q Learning은 예외) |

### 5. Q Learning 핵심 공식

$$Q'(S_t, A_t) = Q(S_t, A_t) + \alpha\{R_t + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)\}$$

- **Off-Policy이지만 중요도 샘플링 불필요** ← 가장 중요한 특징
- 다음 행동을 max Q값으로 선택 → π 불필요
- Bellman optimality equation 기반

### 6. Importance Sampling 원리

$$\rho = \frac{\pi(A|S)}{b(A|S)}, \quad \mathbb{E}_\pi[x] \approx \frac{1}{n}\sum \rho(x^{(i)})x^{(i)}$$

- b 분포에서 샘플링한 데이터로 π 분포의 기댓값 계산
- π와 b가 비슷할수록 분산 작음 (안정적)

---

## ❓ 퀴즈

**Q1.** 다음 중 TD 법의 특징으로 **옳지 않은** 것은?

① 에피소드가 끝날 때까지 기다리지 않고 매 스텝 업데이트한다.
② 환경 모델(전이 확률) 없이도 학습할 수 있다.
③ 일회성 과제에는 적용 가능하지만 지속성 과제에는 적용할 수 없다.
④ Bootstrap을 통해 가치함수를 순차적으로 갱신한다.

> [!done]- 정답
> **③** TD법은 에피소드가 끝나지 않아도 매 스텝 학습하므로, 일회성 과제(O)와 **지속성 과제(O) 모두** 적용 가능하다. MC법이 지속성 과제에 적용 불가하다.

---

**Q2.** SARSA와 Q Learning의 TD Target 수식을 각각 쓰고, 가장 큰 차이점을 설명하라.

> [!done]- 정답
> - **SARSA**: TD Target = $R_t + \gamma Q(S_{t+1}, A_{t+1})$ → 다음 행동 $A_{t+1}$을 정책(π 또는 b)에서 샘플링
> - **Q Learning**: TD Target = $R_t + \gamma \max_a Q(S_{t+1}, a)$ → 다음 행동을 Q값이 최대인 행동으로 결정
>
> **핵심 차이**: Q Learning은 다음 행동을 max 연산으로 결정하므로 중요도 샘플링이 불필요하고 Off-Policy 학습이 안정적이다.

---

**Q3.** On-Policy와 Off-Policy를 구분하는 핵심 기준은 무엇인가?

> [!done]- 정답
> **대상 정책(target policy, π)과 행동 정책(behavior policy, b)의 동일 여부**
> - On-Policy: π = b (자신이 직접 경험한 데이터로 자신의 정책을 개선)
> - Off-Policy: π ≠ b (행동 정책으로 수집한 데이터로 대상 정책을 개선, 중요도 샘플링 필요)

---

**Q4.** 다음 빈칸을 채우시오.
"Off-Policy SARSA에서 행동 정책(b)으로 수집한 데이터로 대상 정책(π)의 Q값을 업데이트할 때, TD Target에 ( )을(를) 곱해 보정해야 한다. 이 값은 ( ) / ( )으로 계산된다."

> [!done]- 정답
> TD Target에 **중요도 비율 ρ(rho)**를 곱해 보정한다.
> $$\rho = \frac{\pi(A_{t+1}|S_{t+1})}{b(A_{t+1}|S_{t+1})}$$
> 대상 정책의 행동 확률 / 행동 정책의 행동 확률

---

**Q5.** Q Learning이 강화학습에서 중요한 이유를 역사적 맥락을 포함하여 서술하시오.

> [!done]- 정답
> Q Learning은 1989년 Watkins & Dayan에 의해 제안된 Off-Policy TD 알고리즘으로, Q함수를 효율적이고 안정적으로 갱신하며 최적 정책을 학습한다. 중요도 샘플링 없이도 Off-Policy 학습이 가능해 안정적이다. 2013년 DeepMind가 Q Learning에 딥러닝을 결합한 DQN(Deep Q-Network)을 발표하여 Atari 게임에서 인간 수준의 성능을 달성하면서, 강화학습의 실용적 가능성을 증명했다. 현재 Double DQN, Dueling DQN, Rainbow DQN 등 수많은 변형이 존재하며, 강화학습 연구의 핵심 기반 알고리즘으로 사용되고 있다.

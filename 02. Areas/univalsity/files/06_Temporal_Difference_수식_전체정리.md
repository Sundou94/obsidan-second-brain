# 06. Temporal Difference Method — 수식 전체 정리

> **출처**: 06-Temporal_Difference_Method.pdf | Prof. Tae-Hyoung Park, CBNU
> ※ MDP 기초 수식은 `03_MDP_수식_전체정리.md`, DP는 `04`, MC는 `05` 참조

---

## 목차

- [[#1. Policy Evaluation — DP vs MC vs TD 비교 (p.3–4)]]
- [[#2. TD 법 — Policy Evaluation 수식 (p.4)]]
- [[#3. SARSA — On-Policy TD Control (p.8–9)]]
- [[#4. On-Policy vs Off-Policy (p.11)]]
- [[#5. Off-Policy SARSA & Importance Sampling (p.13–15)]]
- [[#6. Q-Learning (p.17–19)]]

---

## 1. Policy Evaluation — DP vs MC vs TD 비교 (p.3–4)

> 📄 **원본 슬라이드**: p.3–4

---

### 수식 1-1. Return 정의

$$G_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \cdots = R_t + \gamma G_{t+1}$$

---

### 수식 1-2. 세 가지 Policy Evaluation 방법 비교

**DP 법 (Bootstrapping + 모델 기반)**:

$$V'_\pi(s) = \sum_{a, s'} \pi(a|s)\, p(s'|s, a)\left\{r(s, a, s') + \gamma V_\pi(s')\right\}$$

**MC 법 (샘플 기반, 에피소드 종료 후)**:

$$V'_\pi(S_t) = V_\pi(S_t) + \alpha\left\{G_t - V_\pi(S_t)\right\}$$

**TD 법 (샘플 기반, 매 스텝 업데이트)**:

$$\boxed{V'_\pi(S_t) = V_\pi(S_t) + \alpha\left\{R_t + \gamma V_\pi(S_{t+1}) - V_\pi(S_t)\right\}}$$

| 방법 | 업데이트 기준 | 모델 필요 | 에피소드 종료 필요 |
|------|-------------|---------|----------------|
| DP | 다음 상태의 가치 (bootstrap) | **O** | X |
| MC | 실제 Return $G_t$ (에피소드 전체) | X | **O** |
| TD | 다음 한 스텝만의 추정 (bootstrap + sampling) | X | X |

> **수식의 의미**: TD는 MC처럼 샘플 데이터만 사용하면서도, DP처럼 다음 상태의 추정값으로 현재를 업데이트(bootstrapping)한다. **TD = MC의 샘플링 + DP의 Bootstrap의 결합**이다.

---

## 2. TD 법 — Policy Evaluation 수식 (p.4)

> 📄 **원본 슬라이드**: p.4

---

### 수식 2-1. TD 업데이트 규칙 (TD(0))

$$\boxed{V'_\pi(S_t) = V_\pi(S_t) + \alpha\left\{\underbrace{R_t + \gamma V_\pi(S_{t+1})}_{\text{TD Target}} - V_\pi(S_t)\right\}}$$

| 기호 | 의미 |
|------|------|
| $V_\pi(S_t)$ | 현재 상태 $S_t$에서 목표에 도달 시 얻을 수 있는 수익 **추정치** |
| $V_\pi(S_{t+1})$ | 다음 상태 $S_{t+1}$에서 목표에 도달 시 얻을 수 있는 수익 **추정치** |
| $R_t + \gamma V_\pi(S_{t+1})$ | **TD Target** — 1스텝 실제 보상 + 다음 상태의 할인 추정값 |
| $R_t + \gamma V_\pi(S_{t+1}) - V_\pi(S_t)$ | **TD Error ($\delta_t$)** — 목표와 현재 추정의 차이 |
| $\alpha$ | 학습률 (learning rate) |

> **수식의 의미**: MC 업데이트 $V'(S_t) = V(S_t) + \alpha(G_t - V(S_t))$에서 실제 Return $G_t$ 대신 **1스텝 추정 $R_t + \gamma V(S_{t+1})$** 을 사용한다. 에피소드가 끝날 때까지 기다리지 않고 **매 스텝마다** 가치를 갱신할 수 있다.

---

### 수식 2-2. MC Target vs TD Target 비교

| | 목표값 (Target) | 샘플 필요 범위 |
|--|----------------|--------------|
| **MC** | $G_t = R_t + \gamma R_{t+1} + \cdots$ (실제 Return) | 현재 → 에피소드 종료 |
| **TD** | $R_t + \gamma V_\pi(S_{t+1})$ (1스텝 추정) | 현재 → **다음 스텝** |

> **핵심 차이**: TD는 MC 대비 데이터 **변동성(Variance)이 낮고** 편향(Bias)이 다소 높다. 실제로는 더 빠르게 수렴하는 경향이 있다.

---

## 3. SARSA — On-Policy TD Control (p.8–9)

> 📄 **원본 슬라이드**: p.8–9

---

### 수식 3-1. Action-Value Function TD 업데이트

State-value 업데이트에서 Action-value 업데이트로 확장:

$$\boxed{Q'_\pi(S_t, A_t) = Q_\pi(S_t, A_t) + \alpha\left\{R_t + \gamma Q_\pi(S_{t+1}, A_{t+1}) - Q_\pi(S_t, A_t)\right\}}$$

| 기호 | 의미 |
|------|------|
| $Q_\pi(S_t, A_t)$ | 현재 상태 $S_t$에서 행동 $A_t$를 취할 때의 Q값 추정치 |
| $Q_\pi(S_{t+1}, A_{t+1})$ | 다음 상태 $S_{t+1}$에서 다음 행동 $A_{t+1}$을 취할 때의 Q값 추정치 |
| $R_t + \gamma Q_\pi(S_{t+1}, A_{t+1})$ | **SARSA의 TD Target** |
| $(S_t, A_t, R_t, S_{t+1}, A_{t+1})$ | SARSA가 사용하는 5개 요소 → **S.A.R.S.A.** |

> **수식의 의미**: 다음 행동 $A_{t+1}$을 **현재 정책 $\pi$에서 샘플링**하여 Q값을 업데이트한다. 업데이트에 $(S, A, R, S', A')$ 5개 요소가 필요하다는 점에서 **SARSA**라는 이름이 붙었다.

---

### 수식 3-2. SARSA Policy Update (ε-Greedy)

Q 업데이트 후 정책 갱신:

$$\pi'(a|S_t) = \begin{cases} \arg\max_a Q_\pi(S_t, a) & (1 - \varepsilon \text{ 확률}) \\ \text{무작위 행동} & (\varepsilon \text{ 확률}) \end{cases}$$

| 기호 | 의미 |
|------|------|
| $\varepsilon$ | 탐색 확률 |
| $1 - \varepsilon$ | 활용 확률 (Q값이 최대인 행동 선택) |

> **수식의 의미**: SARSA는 Q 갱신과 동시에 ε-Greedy로 정책을 개선한다. 행동 선택(샘플링)과 정책 개선이 동일한 정책 $\pi$로 이루어지므로 **On-Policy** 방법이다.

---

## 4. On-Policy vs Off-Policy (p.11)

> 📄 **원본 슬라이드**: p.11

---

### 개념 정리

| | On-Policy | Off-Policy |
|-|-----------|------------|
| 대상 정책 (target policy) $\pi$ | 행동 정책과 **동일** | 행동 정책과 **다름** |
| 행동 정책 (behavior policy) $b$ | = $\pi$ | ≠ $\pi$ |
| 탐색 방식 | 자신이 직접 탐색 | 다른 에이전트/정책의 경험 활용 |
| 대표 알고리즘 | SARSA | Q-Learning, Off-Policy SARSA |

> **On-Policy**: 자신이 쌓은 경험으로 자신의 정책을 개선한다.
> **Off-Policy**: 다른 정책(행동 정책 $b$)에서 생성한 데이터로 대상 정책 $\pi$를 개선한다. 더 나은 탐색이 가능하지만 **중요도 샘플링(Importance Sampling)** 이 필요할 수 있다.

---

## 5. Off-Policy SARSA & Importance Sampling (p.13–15)

> 📄 **원본 슬라이드**: p.13–15

---

### 수식 5-1. Off-Policy SARSA 업데이트 (중요도 샘플링 적용)

행동 정책 $b$에서 샘플링된 $A_{t+1}$로 대상 정책 $\pi$의 Q를 갱신:

$$Q'_\pi(S_t, A_t) = Q_\pi(S_t, A_t) + \alpha\left\{\rho\left(R_t + \gamma Q_\pi(S_{t+1}, A_{t+1})\right) - Q_\pi(S_t, A_t)\right\}$$

중요도 샘플링 비율:

$$\rho = \frac{\pi(A_{t+1}|S_{t+1})}{b(A_{t+1}|S_{t+1})}$$

| 기호 | 의미 |
|------|------|
| $\pi$ | **대상 정책 (target policy)** — Greedy로 개선하는 정책 |
| $b$ | **행동 정책 (behavior policy)** — 실제로 행동을 선택하는 정책 (ε-Greedy) |
| $\rho$ | **중요도 샘플링 비율 (Importance Sampling Ratio)** |
| $\pi(A_{t+1}\|S_{t+1})$ | 대상 정책에서 다음 행동을 선택할 확률 |
| $b(A_{t+1}\|S_{t+1})$ | 행동 정책에서 다음 행동을 선택할 확률 |

> **수식의 의미**: 행동 정책 $b$에서 샘플링한 $A_{t+1}$은 대상 정책 $\pi$에서 나온 것이 아니므로, 그대로 사용하면 편향된 업데이트가 된다. $\rho$를 곱해 **행동 정책의 확률 분포 차이를 보정**한다.

---

### 수식 5-2. Importance Sampling 원리

확률분포 $\pi$의 기댓값을 다른 분포 $b$의 샘플로 추정:

$$\mathbb{E}_\pi[x] = \sum_x x\, \pi(x) = \sum_x x \frac{\pi(x)}{b(x)} b(x) = \mathbb{E}_b\left[x \frac{\pi(x)}{b(x)}\right]$$

실용적 추정 (샘플 $x^{(i)} \sim b$, $i = 1, \ldots, n$):

$$\mathbb{E}_\pi[x] \approx \frac{\rho(x^{(1)})x^{(1)} + \cdots + \rho(x^{(n)})x^{(n)}}{n}, \quad \rho(x) = \frac{\pi(x)}{b(x)}$$

| 기호 | 의미 |
|------|------|
| $\mathbb{E}_\pi[x]$ | 확률분포 $\pi$ 하에서 $x$의 기댓값 |
| $\mathbb{E}_b[\cdot]$ | 확률분포 $b$ 하에서의 기댓값 |
| $\rho(x) = \frac{\pi(x)}{b(x)}$ | 중요도 가중치 |
| $x^{(i)} \sim b$ | 분포 $b$에서 샘플링 |

> **수식의 의미**: 분포 $\pi$에서 직접 샘플링하기 어려울 때, 대신 $b$에서 샘플링하고 가중치 $\rho$를 곱해 $\pi$의 기댓값을 추정한다. 두 분포가 유사할수록 $\rho \approx 1$이 되어 추정의 분산이 작아진다.

---

## 6. Q-Learning (p.17–19)

> 📄 **원본 슬라이드**: p.17–19

---

### 수식 6-1. SARSA vs Q-Learning 핵심 비교

**SARSA (On-Policy / Bellman Equation)**:

$$Q'_\pi(S_t, A_t) = Q_\pi(S_t, A_t) + \alpha\left\{R_t + \gamma Q_\pi(S_{t+1}, A_{t+1}) - Q_\pi(S_t, A_t)\right\}$$

**Q-Learning (Off-Policy / Bellman Optimality Equation)**:

$$\boxed{Q'(S_t, A_t) = Q(S_t, A_t) + \alpha\left\{R_t + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t)\right\}}$$

| | SARSA | Q-Learning |
|-|-------|------------|
| TD Target | $R_t + \gamma Q(S_{t+1}, A_{t+1})$ | $R_t + \gamma \max_a Q(S_{t+1}, a)$ |
| 기반 방정식 | Bellman Equation | Bellman **Optimality** Equation |
| $A_{t+1}$ 선택 | 현재 정책 $\pi$(또는 $b$)에서 **샘플링** | **MAX** — 최선 행동을 직접 취함 |
| On/Off-Policy | On-Policy | **Off-Policy** |
| 중요도 샘플링 | Off-policy 시 필요 | **불필요** |

> **수식의 의미**: Q-Learning에서는 다음 행동 $A_{t+1}$을 실제 정책에서 샘플링하지 않고, **$\max_a Q(S_{t+1}, a)$로 최선의 행동을 직접 선택**한다. 이 덕분에 중요도 샘플링이 필요 없고, Q 업데이트가 더 안정적이며, 최적 정책으로 직접 수렴한다.

---

### 수식 6-2. Q-Learning TD Error (TD Target 상세)

$$\delta_t = R_t + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)$$

$$Q'(S_t, A_t) = Q(S_t, A_t) + \alpha\, \delta_t$$

| 기호 | 의미 |
|------|------|
| $\delta_t$ | **TD Error** — Q-Learning 업데이트의 핵심 신호 |
| $R_t$ | 현재 스텝 즉각 보상 |
| $\gamma \max_a Q(S_{t+1}, a)$ | 다음 상태에서 최선 행동을 취할 때의 할인 Q값 |
| $Q(S_t, A_t)$ | 현재 추정값 |

> **수식의 의미**: TD Error $\delta_t$가 양수이면 현재 Q값이 과소 추정된 것이므로 Q값을 높이고, 음수이면 과대 추정이므로 낮춘다. $\alpha$는 얼마나 빠르게 업데이트할지를 조절한다.

---

### 수식 6-3. Q-Learning 행동 정책 업데이트 (ε-Greedy)

Q 업데이트 후 행동 정책 갱신:

$$b(a|s) \leftarrow \text{greedy\_probs}(Q, s, \varepsilon)$$

$$b(a|s) = \begin{cases} 1 - \varepsilon + \frac{\varepsilon}{|\mathcal{A}|}, & a = \arg\max_a Q(s, a) \\ \frac{\varepsilon}{|\mathcal{A}|}, & \text{otherwise} \end{cases}$$

> **수식의 의미**: Q-Learning에서 **대상 정책**은 Greedy (exploit), **행동 정책**은 ε-Greedy (explore)로 분리된다. 대상 정책을 별도로 유지하지 않아도 되는 이유는 $\max_a Q$가 이미 Greedy 선택을 내포하기 때문이다.

---

### 수식 6-4. Q-Learning 요약

- TD 법으로 Q함수 평가 → MC처럼 샘플링 기반, 현재와 다음 정보만 사용
- Bellman Optimality Equation 기반 → 최적 정책으로 직접 수렴
- Off-Policy이지만 중요도 샘플링 불필요 → Q 업데이트 안정적
- 역사적 흐름:
  - 1989: Q-Learning 등장 (Watkins & Dayan)
  - 2013: DQN (Deep Q-Network) — Q-table 대신 신경망으로 Q값 근사 (DeepMind)
  - 이후: Double DQN, Dueling DQN, Rainbow DQN 등 다양한 변형 발전

---

## 수식 전체 요약표

| 번호 | 수식 | 의미 |
|------|------|------|
| 1-1 | $G_t = R_t + \gamma G_{t+1}$ | Return 재귀 정의 |
| 1-2 | $V'(s) = \sum_{a,s'}\pi p\{r+\gamma V(s')\}$ | DP Policy Evaluation |
| 1-3 | $V'(S_t) = V(S_t) + \alpha(G_t - V(S_t))$ | MC Policy Evaluation |
| 2-1 | $V'(S_t) = V(S_t) + \alpha\{R_t + \gamma V(S_{t+1}) - V(S_t)\}$ | **TD(0) Policy Evaluation** |
| 3-1 | $Q'(S_t,A_t) = Q(S_t,A_t) + \alpha\{R_t + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)\}$ | **SARSA 업데이트** |
| 3-2 | $\pi'(a\|S_t) = \arg\max Q$ with prob $1-\varepsilon$ | SARSA ε-Greedy Policy Update |
| 5-1 | $Q'(S_t,A_t) = Q(S_t,A_t) + \alpha\{\rho(R_t + \gamma Q(S_{t+1},A_{t+1})) - Q(S_t,A_t)\}$ | Off-Policy SARSA 업데이트 |
| 5-2 | $\rho = \frac{\pi(A_{t+1}\|S_{t+1})}{b(A_{t+1}\|S_{t+1})}$ | 중요도 샘플링 비율 |
| 5-3 | $\mathbb{E}_\pi[x] \approx \frac{1}{n}\sum \rho(x^{(i)}) x^{(i)}$ | 중요도 샘플링 기댓값 추정 |
| 6-1 | $Q'(S_t,A_t) = Q(S_t,A_t) + \alpha\{R_t + \gamma \max_a Q(S_{t+1},a) - Q(S_t,A_t)\}$ | **Q-Learning 업데이트** |
| 6-2 | $\delta_t = R_t + \gamma \max_a Q(S_{t+1},a) - Q(S_t,A_t)$ | Q-Learning TD Error |

---

## 핵심 기호 사전

| 기호 | 이름 | 의미 |
|------|------|------|
| $\delta_t$ | TD Error | TD Target과 현재 추정값의 차이 |
| $\pi$ | 대상 정책 (target policy) | Off-Policy에서 개선하고자 하는 정책 |
| $b$ | 행동 정책 (behavior policy) | Off-Policy에서 실제로 행동을 선택하는 정책 |
| $\rho$ | 중요도 샘플링 비율 | $\pi(a\|s) / b(a\|s)$ — 두 정책의 확률 비율 |
| $\alpha$ | 학습률 | 업데이트 크기를 조절하는 하이퍼파라미터 |
| $\varepsilon$ | 탐색률 | ε-Greedy에서 무작위 탐색 확률 |
| SARSA | S.A.R.S.A. | $(S_t, A_t, R_t, S_{t+1}, A_{t+1})$ 5요소를 사용하는 TD 알고리즘 |
| Q-Learning | Q학습 | $\max_a Q$ 기반 Off-Policy TD 알고리즘 |
| Bootstrap | 부트스트랩 | 다음 상태의 추정값으로 현재 값을 갱신하는 방식 |
| On-Policy | 온폴리시 | 행동 정책 = 대상 정책 |
| Off-Policy | 오프폴리시 | 행동 정책 ≠ 대상 정책 |

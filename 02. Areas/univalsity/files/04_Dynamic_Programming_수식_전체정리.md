# 04. Dynamic Programming — 수식 전체 정리

> **출처**: 04-Dynamic_Programming.pdf | Prof. Tae-Hyoung Park, CBNU
> ※ MDP 기본 수식은 `03_MDP_수식_전체정리.md` 참조

---

## 목차

- [[#1. MDP 요약 수식 (p.2)]]
- [[#2. Iterative Bellman Equation (p.6–7)]]
- [[#3. Policy Evaluation 수식 (p.8)]]
- [[#4. Bellman Optimality Equation (p.24)]]
- [[#5. Policy Improvement Theorem (p.25)]]
- [[#6. Policy Iteration Method (p.26)]]
- [[#7. Value Iteration Method (p.34–35)]]
- [[#8. Policy Iteration vs Value Iteration 비교]]

---

## 1. MDP 요약 수식 (p.2)

> 📄 **원본 슬라이드**: p.2

---

### 수식 1-1. Bellman 방정식 (State-Value Function)

$$v_\pi(s) = \sum_{a, s'} \pi(a|s)\, p(s'|s, a)\left\{r(s, a, s') + \gamma v_\pi(s')\right\}$$

| 기호 | 의미 |
|------|------|
| $v_\pi(s)$ | 정책 $\pi$ 하에서 상태 $s$의 가치 (state-value) |
| $\sum_{a, s'}$ | 모든 행동 $a$와 다음 상태 $s'$의 조합에 대해 합산 |
| $\pi(a\|s)$ | 상태 $s$에서 행동 $a$를 선택할 확률 (정책) |
| $p(s'\|s, a)$ | 상태 $s$에서 행동 $a$를 취했을 때 다음 상태 $s'$로 전이될 확률 |
| $r(s, a, s')$ | 상태 $s$에서 행동 $a$를 취해 $s'$로 전이될 때 받는 보상 |
| $\gamma$ | 할인율 ($0 \leq \gamma \leq 1$) |
| $v_\pi(s')$ | 다음 상태 $s'$의 가치 |

> **수식의 의미**: 현재 상태 $s$의 가치는 **취할 수 있는 모든 행동과 도달할 수 있는 모든 다음 상태**를 고려하여, 즉각 보상과 다음 상태의 할인 가치의 합산으로 표현된다. 이 식은 현재와 다음 상태의 가치 간 **재귀적 관계**를 나타낸다.

---

### 수식 1-2. Bellman 방정식 (Action-Value Function)

$$q_\pi(s, a) = \sum_{s'} p(s'|s, a)\left\{r(s, a, s') + \gamma \sum_{a'} \pi(a'|s')\, q_\pi(s', a')\right\}$$

| 기호 | 의미 |
|------|------|
| $q_\pi(s, a)$ | 정책 $\pi$ 하에서 상태 $s$에서 행동 $a$를 취할 때의 가치 (action-value) |
| $\sum_{s'}$ | 다음 상태 $s'$ 전체에 대한 합산 |
| $\sum_{a'} \pi(a'\|s') q_\pi(s', a')$ | 다음 상태 $s'$에서 정책 $\pi$를 따를 때의 기대 행동 가치 = $v_\pi(s')$ |

> **수식의 의미**: 행동 가치 함수는 현재 행동 이후 다음 상태에서 정책 $\pi$를 따라 얻을 수 있는 기대 보상으로 표현된다. 내부의 합산 $\sum_{a'} \pi(a'|s') q_\pi(s', a')$는 곧 $v_\pi(s')$와 같다.

---

### 수식 1-3. 최적 상태 가치 / 행동 가치

$$v_*(s) = \max_\pi v_\pi(s), \qquad q_*(s, a) = \max_\pi q_\pi(s, a)$$

| 기호 | 의미 |
|------|------|
| $v_*(s)$ | 상태 $s$에서 **모든 정책 중 달성 가능한 최대 가치** |
| $q_*(s, a)$ | 상태 $s$에서 행동 $a$를 취할 때 **모든 정책 중 달성 가능한 최대 가치** |
| $\max_\pi$ | 존재하는 모든 정책 $\pi$에 대한 최대값 |

> **수식의 의미**: 최적 가치 함수는 모든 가능한 정책 중 가장 좋은 성능을 내는 정책을 따랐을 때 달성할 수 있는 가치다. RL의 궁극적 목표는 이 최적 가치 함수 또는 그것을 달성하는 최적 정책을 찾는 것이다.

---

### 수식 1-4. 최적 정책 (결정론적)

$$\pi_*(a|s) = \begin{cases} 1, & \text{if } a = \arg\max_{a \in \mathcal{A}} q_*(s, a) \\ 0, & \text{otherwise} \end{cases}$$

$$\mu_*(s) = \arg\max_a q_*(s, a) \qquad \text{(optimal action)}$$

| 기호 | 의미 |
|------|------|
| $\arg\max_a q_*(s, a)$ | $q_*(s, a)$를 최대화하는 **행동 $a$ 자체** |
| $\mu_*(s)$ | 상태 $s$에서의 **최적 행동** (결정론적 정책의 압축 표현) |

> **수식의 의미**: 최적 행동은 해당 상태에서 $q_*$가 가장 큰 행동을 선택하는 것이며, 이를 **Greedy selection** 이라 한다.

---

## 2. Iterative Bellman Equation (p.6–7)

> 📄 **원본 슬라이드**: p.6–7

---

### 수식 2-1. Iterative Bellman Equation (일반형)

$$\boxed{V_{k+1}(s) = \sum_{a, s'} \pi(a|s)\, p(s'|s, a)\left\{r(s, a, s') + \gamma V_k(s')\right\}}$$

$$V_0(s) \to V_1(s) \to \cdots \to V_k(s) \to V_{k+1}(s) \to \cdots \approx v_\pi(s)$$

| 기호 | 의미 |
|------|------|
| $V_k(s)$ | $k$번째로 갱신된 state-value의 **추정치** (실제 가치 $v_\pi(s)$와 다름) |
| $V_{k+1}(s)$ | $k+1$번째 갱신 추정치 |
| $V_k(s')$ | 이전 추정치 $V_k$로부터 계산한 다음 상태의 가치 |

> **수식의 의미**: 연립방정식을 직접 푸는 대신, 추정치 $V_k(s')$로부터 다음 추정치 $V_{k+1}(s)$를 개선하는 **반복(iteration) 방식**으로 접근한다. 이 반복 업데이트를 **Bootstrapping** 이라 한다. $k \to \infty$이면 $V_k(s) \to v_\pi(s)$로 수렴한다.

---

### 수식 2-2. Iterative Bellman Equation (결정론적 전이)

결정론적 상태 전이의 경우 $p(s'|s, a)$는:

$$p(s'|s, a) = \begin{cases} 1, & \text{if } s' = f(s, a) \\ 0, & \text{if } s' \neq f(s, a) \end{cases}$$

따라서 Iterative Bellman Equation이 단순화된다:

$$\boxed{V_{k+1}(s) = \sum_a \pi(a|s)\left\{r(s, a, s') + \gamma V_k(s')\right\}} \qquad s' = f(s, a)$$

| 기호 | 의미 |
|------|------|
| $f(s, a)$ | 상태 $s$에서 행동 $a$를 취했을 때 결정론적으로 도달하는 다음 상태 |
| $s' = f(s, a)$ | 상태 전이가 확률적이지 않고 **하나의 결과로 확정** |

> **수식의 의미**: 확률론적 전이의 경우 $\sum_{s'}$를 통해 모든 가능한 다음 상태를 고려해야 하지만, 결정론적 전이에서는 다음 상태가 하나로 고정되므로 합산이 사라진다.

---

## 3. Policy Evaluation 수식 (p.8)

> 📄 **원본 슬라이드**: p.8

---

### 수식 3-1. Iterative Policy Evaluation 알고리즘

반복 업데이트 방정식:

$$V(s) \leftarrow \sum_a \pi(a|s) \sum_{s', r} p(s', r|s, a)\left[r + \gamma V(s')\right]$$

수렴 조건:

$$\Delta \leftarrow \max\left(\Delta,\, |v - V(s)|\right) < \theta \quad \text{(a small positive number)}$$

| 기호 | 의미 |
|------|------|
| $V(s) \leftarrow \cdots$ | 현재 추정치를 새 값으로 **덮어쓰기 (in-place update)** |
| $\Delta$ | 이번 갱신에서 발생한 **최대 변화량** |
| $\theta$ | 수렴 판단을 위한 **임계값 (threshold)** |
| $v$ | 갱신 전 이전 값 ($v \leftarrow V(s)$로 저장해둔 값) |

> **수식의 의미**: 모든 상태 $s$에 대해 Bellman 방정식으로 가치를 반복 갱신하며, 갱신량의 최대값 $\Delta$가 임계값 $\theta$ 이하로 떨어질 때까지 반복한다. 이것이 **Iterative Policy Evaluation** 알고리즘이다.

---

### 수식 3-2. In-place Update (동시 갱신 vs 즉시 반영)

**동시 갱신 (Two-array method)**:
$$V_\text{new}(s) \leftarrow f(V_\text{old})$$

**즉시 반영 (In-place method)**:
$$V(s) \leftarrow f(V) \quad \text{(이미 갱신된 값 즉시 활용)}$$

| 방식 | 특징 |
|------|------|
| Two-array | $V_k$와 $V_{k+1}$를 분리 보관, 갱신 순서 무관 |
| In-place | 갱신된 값을 즉시 다음 계산에 활용, **일반적으로 수렴 속도 빠름** |

> **수식의 의미**: In-place 방식은 이미 갱신된 이웃 상태 값을 즉시 활용하므로 Two-array 방식보다 더 빠르게 수렴하는 경향이 있다. (슬라이드 예제: Two-array 76회 vs In-place 60회 갱신으로 수렴)

---

## 4. Bellman Optimality Equation (p.24)

> 📄 **원본 슬라이드**: p.24

---

### 수식 4-1. Bellman Optimality Equation (State-Value)

$$v_*(s) = \max_a \sum_{s'} p(s'|s, a)\left\{r(s, a, s') + \gamma v_*(s')\right\}$$

| 기호 | 의미 |
|------|------|
| $v_*(s)$ | 상태 $s$의 **최적 가치** |
| $\max_a$ | 모든 가능한 행동 $a$ 중 **최대값을 선택** |
| $\sum_{s'} p(s'\|s, a)\{\cdots\}$ | 행동 $a$를 취했을 때 기대되는 다음 상태 가치의 합 |

> **수식의 의미**: 최적 가치 함수는 최적 행동(max를 달성하는 행동)을 따랐을 때 얻는 기대 보상으로 재귀적으로 표현된다. Bellman 방정식에서 $\sum_a \pi(a|s)$ 대신 $\max_a$로 바뀐 것이 핵심이다.

---

### 수식 4-2. Bellman Optimality Equation (Action-Value)

$$q_*(s, a) = \sum_{s'} p(s'|s, a)\left\{r(s, a, s') + \gamma \max_{a'} q_*(s', a')\right\}$$

| 기호 | 의미 |
|------|------|
| $q_*(s, a)$ | 상태 $s$에서 행동 $a$를 취할 때의 **최적 행동 가치** |
| $\max_{a'} q_*(s', a')$ | 다음 상태 $s'$에서 **최선의 행동을 선택했을 때의 가치** |

> **수식의 의미**: 최적 행동 가치는 현재 행동 $a$ 이후, 다음 상태 $s'$에서 최선의 행동을 계속 선택한다고 가정할 때의 기대 보상이다. Q-Learning 업데이트 규칙의 이론적 기반이다.

---

### 수식 4-3. 최적 행동 도출 (결정론적 전이)

$$\mu_*(s) = \arg\max_a q_*(s, a) = \arg\max_a \sum_{s'} p(s'|s, a)\left\{r(s, a, s') + \gamma v_\pi(s')\right\}$$

결정론적 전이의 경우:

$$\mu_*(s) = \arg\max_a \left\{r(s, a, s') + \gamma v_\pi(s')\right\} \qquad s' = f(s, a)$$

| 기호 | 의미 |
|------|------|
| $\mu_*(s)$ | 상태 $s$에서의 **최적 행동 (optimal action)** |
| $\arg\max_a$ | 값을 최대화하는 행동 $a$ 자체를 반환 |

> **수식의 의미**: $v_\pi(s)$를 알고 있다면, 각 상태에서 한 스텝 앞을 내다보는 Greedy 탐색으로 최적 행동을 찾을 수 있다.

---

## 5. Policy Improvement Theorem (p.25)

> 📄 **원본 슬라이드**: p.25

---

### 수식 5-1. Policy Improvement Theorem

결정론적 정책 $\pi$와 $\pi'$의 쌍에 대해, 모든 $s \in \mathcal{S}$에서:

$$q_\pi(s, \mu'(s)) \geq v_\pi(s)$$

이면:

$$v_{\pi'}(s) \geq v_\pi(s)$$

| 기호 | 의미 |
|------|------|
| $\pi$ | 현재 정책 |
| $\pi'$ | 새로운(개선된) 정책 |
| $\mu'(s)$ | 정책 $\pi'$에서 상태 $s$의 행동 |
| $q_\pi(s, \mu'(s)) \geq v_\pi(s)$ | 새 정책의 행동이 현재 정책보다 기대 가치가 크거나 같음 |
| $v_{\pi'}(s) \geq v_\pi(s)$ | 따라서 새 정책 $\pi'$는 $\pi$보다 좋거나 같다 |

> **수식의 의미**: 어떤 상태에서든 현재 정책보다 더 높은 행동 가치를 내는 행동을 선택하는 새 정책은, **전체 상태 공간에 걸쳐 현재 정책보다 좋거나 같은 성능을 보장**한다. 이것이 Policy Improvement의 이론적 근거다.

---

### 수식 5-2. Policy Improvement 절차

$$v_\pi(s) = \sum_a \pi(a|s)\, q_\pi(s, a)$$

$$q_\pi(s, a) = \sum_{s'} p(s'|s, a)\left\{r(s, a, s') + \gamma v_\pi(s')\right\}$$

최적 정책을 향한 반복:

$$\pi_0 \xrightarrow{E} v_{\pi_0} \xrightarrow{I} \pi_1 \xrightarrow{E} v_{\pi_1} \xrightarrow{I} \pi_2 \xrightarrow{E} \cdots \xrightarrow{I} \pi_* \xrightarrow{E} v_*$$

| 기호 | 의미 |
|------|------|
| $E$ | **Policy Evaluation** — 현재 정책 $\pi_k$에 대한 가치 함수 계산 |
| $I$ | **Policy Improvement** — 현재 가치 함수로부터 Greedy하게 정책 개선 |

> **수식의 의미**: Evaluation(평가)과 Improvement(개선)을 교대로 반복하면 최종적으로 최적 정책 $\pi_*$와 최적 가치 $v_*$에 수렴한다.

---

## 6. Policy Iteration Method (p.26)

> 📄 **원본 슬라이드**: p.26

---

### 수식 6-1. Policy Evaluation 단계

$$V(s) \leftarrow \sum_{s', r} p(s', r|s, \pi(s))\left[r + \gamma V(s')\right], \quad \Delta \leftarrow \max\left(\Delta, |v - V(s)|\right)$$

반복: $\Delta < \theta$ 일 때까지

---

### 수식 6-2. Policy Improvement 단계 (Greedy Selection)

$$\pi(s) \leftarrow \arg\max_a \sum_{s', r} p(s', r|s, a)\left[r + \gamma V(s')\right]$$

결정론적 전이 시:

$$\mu_*(s) = \arg\max_a \left\{r(s, a, s') + \gamma v_\pi(s')\right\}$$

$$\pi(s|\mu_*(s)) = 1$$

| 기호 | 의미 |
|------|------|
| $\arg\max_a$ | 괄호 안의 값을 최대화하는 행동 $a$ |
| $\pi(s) \leftarrow \cdots$ | 정책을 Greedy하게 갱신 |
| policy-stable | 정책이 더 이상 변하지 않으면 **수렴** 판단 |

> **수식의 의미**: Policy Iteration은 (1) 현재 정책에 대한 완전한 Policy Evaluation, (2) 그 가치 함수로부터 Greedy한 Policy Improvement를 반복하는 방법이다. 정책이 더 이상 바뀌지 않으면 최적 정책에 수렴했음을 의미한다.

---

## 7. Value Iteration Method (p.34–35)

> 📄 **원본 슬라이드**: p.34–35

---

### 수식 7-1. Value Iteration Evaluation 단계

Bellman Optimality Equation을 반복 업데이트:

$$\boxed{V'(s) = \max_a \sum_{s'} p(s'|s, a)\left\{r(s, a, s') + \gamma V(s')\right\}}$$

결정론적 전이 시:

$$V'(s) = \max_a \left\{r(s, a, s') + \gamma V(s')\right\} \qquad s' = f(s, a)$$

| 기호 | 의미 |
|------|------|
| $\max_a$ | **정책 없이** 직접 최선의 행동을 선택하며 가치 갱신 |
| $V'(s)$ | 갱신된 최적 가치 추정치 |

> **수식의 의미**: Policy Iteration의 Evaluation 단계에서 $\sum_a \pi(a|s)$로 가중 평균을 내던 것 대신, **$\max_a$로 최선 행동을 직접 선택**하여 가치를 갱신한다. 정책을 명시적으로 유지하지 않아도 된다.

---

### 수식 7-2. Value Iteration Improvement 단계

수렴 후 최적 정책 도출:

$$\mu(s) = \arg\max_a \sum_{s'} p(s'|s, a)\left\{r(s, a, s') + \gamma V(s')\right\}$$

---

### 수식 7-3. Policy Iteration vs Value Iteration 핵심 비교

| | Policy Iteration | Value Iteration |
|-|-----------------|----------------|
| Evaluation 수식 | $V'(s) = \sum_a \pi(a\|s)\{r + \gamma V(s')\}$ | $V'(s) = \max_a\{r + \gamma V(s')\}$ |
| 사용 방정식 | Bellman Equation | Bellman **Optimality** Equation |
| 정책 $\pi$ 필요 여부 | **필요** | **불필요** |
| 수렴 속도 | Evaluation 반복 많음 | 일반적으로 **빠름** |

> **핵심 차이**: Policy Iteration은 완전한 Policy Evaluation 후 Improvement를 반복하는 반면, Value Iteration은 매 스텝마다 Evaluation과 Improvement를 **동시에** 수행한다 (evaluation with improvement).

---

## 8. DP의 한계 및 요약 (p.43)

> 📄 **원본 슬라이드**: p.43

---

### 수식 8-1. DP 방법들의 요약

최적 정책 도달 경로:

$$\pi_0 \xrightarrow{E} v_{\pi_0} \xrightarrow{I} \pi_1 \xrightarrow{E} v_{\pi_1} \xrightarrow{I} \cdots \xrightarrow{I} \pi_* \xrightarrow{E} v_*$$

| 항목 | Policy Iteration | Value Iteration |
|------|-----------------|----------------|
| Policy Evaluation | Bellman Equation | Bellman Optimality Equation |
| Policy Improvement | Greedy selection | Greedy selection |

**DP의 한계**:
- **Curse of Dimensionality**: 상태/행동 차원이 커질수록 계산량이 기하급수적으로 증가
- **환경 모델 필요**: $p(s'|s, a)$와 $r(s, a, s')$ 를 사전에 알고 있어야 함 → 이를 극복하기 위해 Monte Carlo 및 TD 방법 등장

---

## 수식 전체 요약표

| 번호 | 수식 | 의미 |
|------|------|------|
| 1-1 | $v_\pi(s) = \sum_{a,s'} \pi(a\|s)\,p(s'\|s,a)\{r + \gamma v_\pi(s')\}$ | Bellman 방정식 (state-value) |
| 1-2 | $q_\pi(s,a) = \sum_{s'} p(s'\|s,a)\{r + \gamma \sum_{a'}\pi(a'\|s')q_\pi(s',a')\}$ | Bellman 방정식 (action-value) |
| 1-3 | $v_*(s) = \max_\pi v_\pi(s)$ | 최적 state-value |
| 1-4 | $\mu_*(s) = \arg\max_a q_*(s,a)$ | 최적 행동 |
| 2-1 | $V_{k+1}(s) = \sum_{a,s'} \pi(a\|s)p(s'\|s,a)\{r + \gamma V_k(s')\}$ | Iterative Bellman Equation |
| 2-2 | $V_{k+1}(s) = \sum_a \pi(a\|s)\{r + \gamma V_k(s')\}$ | 결정론적 전이의 Iterative Bellman Eq. |
| 3-1 | $\Delta \leftarrow \max(\Delta, \|v - V(s)\|) < \theta$ | 수렴 조건 |
| 4-1 | $v_*(s) = \max_a \sum_{s'} p(s'\|s,a)\{r + \gamma v_*(s')\}$ | Bellman Optimality Eq. (state) |
| 4-2 | $q_*(s,a) = \sum_{s'} p(s'\|s,a)\{r + \gamma \max_{a'} q_*(s',a')\}$ | Bellman Optimality Eq. (action) |
| 4-3 | $\mu_*(s) = \arg\max_a\{r + \gamma v_\pi(s')\}$ | 결정론적 최적 행동 |
| 5-1 | $q_\pi(s, \mu'(s)) \geq v_\pi(s) \Rightarrow v_{\pi'}(s) \geq v_\pi(s)$ | Policy Improvement Theorem |
| 5-2 | $\pi_0 \xrightarrow{E} v_{\pi_0} \xrightarrow{I} \cdots \to \pi_* \xrightarrow{E} v_*$ | 최적 정책 수렴 경로 |
| 6-1 | $\pi(s) \leftarrow \arg\max_a \sum_{s',r} p(s',r\|s,a)[r + \gamma V(s')]$ | Policy Improvement (Greedy) |
| 7-1 | $V'(s) = \max_a\{r + \gamma V(s')\}$ | Value Iteration 갱신 수식 |
| 7-2 | $\mu(s) = \arg\max_a \sum_{s'} p(s'\|s,a)\{r + \gamma V(s')\}$ | Value Iteration 최적 정책 도출 |

---

## 핵심 기호 사전

| 기호 | 이름 | 의미 |
|------|------|------|
| $V_k(s)$ | k번째 가치 추정치 | Iterative 방법에서 $k$번 갱신된 가치 함수의 추정값 |
| $\theta$ | 임계값 (threshold) | Policy Evaluation 수렴 판단 기준 |
| $\Delta$ | 최대 변화량 | 한 번의 갱신 사이클에서 가장 큰 값 변화 |
| $E$ | Evaluation | 현재 정책에 대한 가치 함수 계산 |
| $I$ | Improvement | 현재 가치 함수로부터 정책 개선 |
| $f(s, a)$ | 결정론적 전이 함수 | 상태 $s$에서 행동 $a$ 시 결정론적으로 도달하는 다음 상태 |
| $\mu(s)$ | 결정론적 정책 | 상태 $s$에서 선택할 행동을 직접 반환하는 함수 |
| policy-stable | 정책 안정 | 갱신 후에도 정책이 변하지 않는 상태 = 수렴 |

# 05. Monte Carlo Method — 수식 전체 정리

> **출처**: 05-Monte_Carlo_Method.pdf | Prof. Tae-Hyoung Park, CBNU
> ※ MDP 기본 수식은 `03_MDP_수식_전체정리.md`, DP 수식은 `04_Dynamic_Programming_수식_전체정리.md` 참조

---

## 목차

- [[#1. Monte Carlo 기초 (p.2–5)]]
- [[#2. Monte Carlo Prediction (p.6–7)]]
- [[#3. Policy Evaluation — MC 알고리즘 (p.8–11)]]
- [[#4. Monte Carlo Control — Action-Value Function (p.13–14)]]
- [[#5. ε-Greedy Policy (p.17)]]
- [[#6. Exponential Moving Average (p.18)]]
- [[#7. DP vs MC 비교 요약 (p.23)]]

---

## 1. Monte Carlo 기초 (p.2–5)

> 📄 **원본 슬라이드**: p.2–5

---

### 수식 1-1. 기댓값의 표본 평균 근사 (Sample Mean)

$$V_n = \frac{s_1 + s_2 + \cdots + s_n}{n}$$

| 기호 | 의미 |
|------|------|
| $V_n$ | $n$개의 샘플로 추정한 **기댓값** |
| $s_1, s_2, \ldots, s_n$ | 독립적으로 얻은 **샘플값** |
| $n$ | 샘플 수 |

> **수식의 의미**: Monte Carlo 방법의 핵심 — 분포를 모르더라도 **반복 샘플링을 통해 기댓값을 추정**할 수 있다. 대수의 법칙(Law of Large Numbers)에 의해 $n \to \infty$이면 $V_n \to \mathbb{E}[s]$로 수렴한다.

---

### 수식 1-2. 증분 평균 (Incremental Mean)

$$V_n = V_{n-1} + \frac{1}{n}\left(s_n - V_{n-1}\right)$$

| 기호 | 의미 |
|------|------|
| $V_n$ | $n$번째 샘플까지 고려한 현재 추정값 |
| $V_{n-1}$ | 이전 추정값 |
| $s_n$ | $n$번째 새로운 샘플 |
| $\frac{1}{n}$ | 학습률 (샘플이 쌓일수록 각 샘플의 영향 감소) |
| $s_n - V_{n-1}$ | **예측 오차 (Prediction Error)** — 새 샘플과 현재 추정의 차이 |

> **수식의 의미**: 모든 샘플을 저장하지 않고 **현재 추정값과 새 샘플의 차이만으로 업데이트**한다. 이 형태가 강화학습 전체에서 반복 등장하는 핵심 업데이트 패턴이다. "추정 오차의 일부만큼 추정값을 조정한다"는 직관을 갖는다.

---

## 2. Monte Carlo Prediction (p.6–7)

> 📄 **원본 슬라이드**: p.6–7

---

### 수식 2-1. Monte Carlo 방법에 의한 State-Value 추정

$$V_\pi(s) = \frac{G^{(1)} + G^{(2)} + \cdots + G^{(n)}}{n}$$

| 기호 | 의미 |
|------|------|
| $V_\pi(s)$ | MC 방법으로 추정한 상태 $s$의 가치 |
| $G^{(n)}$ | $n$번째 에피소드에서 상태 $s$를 방문했을 때 얻은 **실제 Return** |
| $n$ | 상태 $s$를 방문한 **에피소드 횟수** |

> **수식의 의미**: 진짜 가치 $v_\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$를 **샘플 Return의 평균으로 근사**한다. DP처럼 환경 모델 $p(s'|s,a)$를 필요로 하지 않고, **실제 경험(에피소드)으로부터 직접 학습**한다.
>
> **DP와의 비교**:
> - DP: $V_{k+1}(s) = \sum_{a,s'} \pi(a|s)\,p(s'|s,a)\{r + \gamma V_k(s')\}$ → Computing (모델 필요)
> - MC: $V_\pi(s) = \frac{1}{n}\sum G^{(n)}$ → Learning (경험 기반)

---

### 수식 2-2. Return의 역방향 재귀 계산 (Efficient)

에피소드 $A \to B \to C \to \text{terminal}$, 보상 $R_0, R_1, R_2$에 대해:

**비효율적 방식 (중복 계산)**:
$$G_A = R_0 + \gamma R_1 + \gamma^2 R_2$$
$$G_B = R_1 + \gamma R_2$$
$$G_C = R_2$$

**효율적 방식 (역방향 계산)**:
$$G_C = R_2$$
$$G_B = R_1 + \gamma G_C$$
$$G_A = R_0 + \gamma G_B$$

| 기호 | 의미 |
|------|------|
| reversed(memory) | 에피소드 데이터를 **역순으로** 순회 |
| $G \leftarrow \gamma G + R$ | 역방향 누적으로 Return 계산 |

> **수식의 의미**: 에피소드 끝부터 역방향으로 $G = R + \gamma G$를 반복하면, 각 상태의 Return을 **중복 계산 없이 단일 패스로** 구할 수 있다. Python 코드에서 `for data in reversed(self.memory):`로 구현된다.

---

## 3. Policy Evaluation — MC 알고리즘 (p.8–11)

> 📄 **원본 슬라이드**: p.8–11

---

### 수식 3-1. MC Policy Evaluation 알고리즘

에피소드 $\pi$ 생성 → 각 상태 $s$의 첫 방문 이후 Return $G$ 수집:

$$V(s) \leftarrow \text{average}(Returns(s))$$

증분 형태:

$$V(s) \leftarrow V(s) + \frac{1}{n_s}\left(G - V(s)\right)$$

| 기호 | 의미 |
|------|------|
| $Returns(s)$ | 상태 $s$를 방문한 각 에피소드에서 얻은 Return들의 리스트 |
| $n_s$ | 상태 $s$를 방문한 **누적 횟수** |
| $G - V(s)$ | 실제 Return과 현재 추정값의 **오차** |

> **수식의 의미**: 에피소드가 종료될 때마다 상태 $s$의 가치 추정을 갱신한다. 새로운 Return이 현재 추정보다 크면 값을 높이고, 작으면 낮춘다. 에피소드를 반복할수록 추정이 실제 $v_\pi(s)$에 수렴한다.

---

## 4. Monte Carlo Control — Action-Value Function (p.13–14)

> 📄 **원본 슬라이드**: p.13–14

---

### 수식 4-1. MC Policy Improvement (Greedy Selection)

$$\mu(s) = \arg\max_a q(s, a)$$

| 기호 | 의미 |
|------|------|
| $\mu(s)$ | 상태 $s$에서 Q함수를 최대화하는 **최적 행동** |
| $q(s, a)$ | 현재까지 추정된 행동 가치 함수 (Action-Value Function) |
| $\arg\max_a$ | Q값이 가장 큰 행동 $a$를 직접 반환 |

**Policy Improvement Theorem 적용**:
$$q_{\pi_k}\left(s, \mu_{k+1}(s)\right) = q_{\pi_k}\left(s, \arg\max_a q_{\pi_k}(s, a)\right) = \max_a q_{\pi_k}(s, a) \geq v_{\pi_k}(s)$$

> **수식의 의미**: 현재 Q함수에 대해 Greedy하게 행동을 선택하면, 새 정책은 현재 정책보다 성능이 반드시 좋거나 같음이 보장된다.

---

### 수식 4-2. Action-Value Function 업데이트 (일반 방식)

$$Q_n(s, a) = \frac{G^{(1)} + G^{(2)} + \cdots + G^{(n)}}{n}$$

| 기호 | 의미 |
|------|------|
| $Q_n(s, a)$ | 상태 $s$에서 행동 $a$를 취한 후 얻은 $n$개 Return의 평균 |
| $G^{(n)}$ | $n$번째 에피소드에서 $(s, a)$를 방문한 후 얻은 Return |

---

### 수식 4-3. Action-Value Function 업데이트 (증분 방식)

$$Q_n(s, a) = Q_{n-1}(s, a) + \frac{1}{n}\left\{G^{(n)} - Q_{n-1}(s, a)\right\}$$

| 기호 | 의미 |
|------|------|
| $Q_{n-1}(s, a)$ | 이전 Q값 추정 |
| $G^{(n)} - Q_{n-1}(s, a)$ | 새 Return과 현재 Q값의 오차 |
| $\frac{1}{n}$ | $n$번째 샘플의 가중치 |

> **수식의 의미**: 모든 Return을 저장하지 않고 **증분 방식**으로 Q값을 업데이트한다. MC Control에서 이 업데이트 후 Greedy 정책 개선을 반복하면 최적 Q함수 $q_*$에 수렴한다.

---

### 수식 4-4. 결정론적 최적 정책 (MC Control 결과)

$$\pi(s|a) = \begin{cases} 1, & a = \mu(s) \\ 0, & \text{otherwise} \end{cases}$$

> **수식의 의미**: MC Control이 수렴하면 각 상태에서 Q값이 가장 높은 행동만을 선택하는 결정론적 최적 정책을 얻는다.

---

## 5. ε-Greedy Policy (p.17)

> 📄 **원본 슬라이드**: p.17

---

### 수식 5-1. ε-Greedy 정책

$$\pi'(a|s) = \begin{cases} \arg\max_a Q_\pi(s, a) & (1 - \varepsilon \text{의 확률}) \\ \text{무작위 행동} & (\varepsilon \text{의 확률}) \end{cases}$$

구체적 확률:

$$\pi'(a|s) = \begin{cases} 1 - \varepsilon + \dfrac{\varepsilon}{|\mathcal{A}|}, & a = \arg\max_a Q(s, a) \\ \dfrac{\varepsilon}{|\mathcal{A}|}, & \text{otherwise} \end{cases}$$

| 기호 | 의미 |
|------|------|
| $\varepsilon$ | 탐색(exploration) 확률, $0 \leq \varepsilon \leq 1$ |
| $1 - \varepsilon$ | 활용(exploitation) 확률 |
| $|\mathcal{A}|$ | 가능한 행동의 총 수 |
| $\frac{\varepsilon}{|\mathcal{A}|}$ | 각 행동에 균등하게 분배되는 탐색 확률 |

> **수식의 의미**: 순수 Greedy 정책($\varepsilon = 0$)은 처음에 방문하지 않은 상태-행동 쌍의 Q값을 절대 갱신할 기회가 없다 (탐색 부재). ε-Greedy는 작은 확률 $\varepsilon$으로 무작위 탐색을 섞어, **Exploration(탐색)과 Exploitation(활용)의 균형**을 맞춘다.
>
> **예시**: $\varepsilon = 0.4$, 행동 수 4, 최적 행동 = 1이면:
> $$\text{action\_probs} = \{0: 0.1,\ 1: 0.7,\ 2: 0.1,\ 3: 0.1\}$$

---

## 6. Exponential Moving Average (p.18)

> 📄 **원본 슬라이드**: p.18

---

### 수식 6-1. 고정 학습률 업데이트 (Exponential Moving Average)

$$Q_n(s, a) = Q_{n-1}(s, a) + \alpha\left\{G^{(n)} - Q_{n-1}(s, a)\right\}$$

| 기호 | 의미 |
|------|------|
| $\alpha$ | 고정 **학습률 (learning rate)**, $0 < \alpha \leq 1$ |
| $G^{(n)} - Q_{n-1}(s, a)$ | TD 오차 유사 항 (새 Return과 현재 추정의 차이) |

**$\frac{1}{n}$ 방식과의 가중치 비교**:

| 방식 | $G^{(1)}, G^{(2)}, \ldots, G^{(n)}$의 가중치 |
|------|------|
| $\frac{1}{n}$ | $\frac{1}{n}, \frac{1}{n}, \ldots, \frac{1}{n}$ (균등) |
| $\alpha$ (고정) | $\alpha(1-\alpha)^{n-1},\ \alpha(1-\alpha)^{n-2},\ \ldots,\ \alpha$ (최신 데이터에 높은 가중치) |

> **수식의 의미**: $\frac{1}{n}$ 방식은 모든 과거 데이터를 동등하게 취급한다. 고정 $\alpha$ 방식은 **최신 데이터에 더 큰 가중치**를 부여하며, 비정상(non-stationary) 환경에 더 적합하다. 가중치가 지수적으로(exponentially) 감소하므로 **Exponential Moving Average** 라고 한다.

---

## 7. DP vs MC 비교 요약 (p.23)

> 📄 **원본 슬라이드**: p.23

---

### 비교표

| 항목 | DP | MC |
|------|----|----|
| Policy Evaluation | $V'(s) = \sum_{a,s'} \pi(a\|s)p(s'\|s,a)\{r + \gamma V(s')\}$ | $V_\pi(s) = \frac{G^{(1)}+\cdots+G^{(n)}}{n}$ |
| Policy Control | Policy Iteration / Value Iteration | $\mu(s) = \arg\max_a Q(s,a)$ |
| 환경 모델 필요 | **O** ($p(s'\|s,a)$, $r(s,a,s')$ 필요) | **X** (샘플만으로 가능) |
| Curse of Dimensionality | **O** (차원 증가 시 계산량 폭발) | **X** |
| 일회성 과제(Episodic) | O | O |
| 지속성 과제(Continuing) | O | **X** (에피소드 종료 필요) |
| Markov Property 필요 | O | X |

> **핵심 차이**: DP는 **환경 모델을 알고 계산(computing)**하는 방식이고, MC는 **실제 경험으로부터 학습(learning)**하는 방식이다. MC의 한계(지속성 과제 불가, 에피소드 종료 후 갱신)를 해결하기 위해 **TD(Temporal Difference) 방법**이 등장한다.

---

## 수식 전체 요약표

| 번호 | 수식 | 의미 |
|------|------|------|
| 1-1 | $V_n = \frac{1}{n}\sum_{i=1}^n s_i$ | 샘플 평균으로 기댓값 추정 |
| 1-2 | $V_n = V_{n-1} + \frac{1}{n}(s_n - V_{n-1})$ | 증분 평균 업데이트 |
| 2-1 | $V_\pi(s) = \frac{G^{(1)}+\cdots+G^{(n)}}{n}$ | MC state-value 추정 |
| 2-2 | $G_t = R_t + \gamma G_{t+1}$ (역방향 계산) | Return 효율적 계산 |
| 3-1 | $V(s) \leftarrow V(s) + \frac{1}{n_s}(G - V(s))$ | MC Policy Evaluation 업데이트 |
| 4-1 | $\mu(s) = \arg\max_a q(s,a)$ | MC Greedy Policy Improvement |
| 4-2 | $Q_n(s,a) = \frac{1}{n}\sum G^{(n)}$ | Action-value 추정 (일반) |
| 4-3 | $Q_n(s,a) = Q_{n-1}(s,a) + \frac{1}{n}(G^{(n)} - Q_{n-1}(s,a))$ | Action-value 업데이트 (증분) |
| 4-4 | $\pi(s\|a) = 1$ if $a = \mu(s)$, else $0$ | 결정론적 최적 정책 |
| 5-1 | $\pi'(a\|s) = \arg\max$ with prob $1-\varepsilon$, random with prob $\varepsilon$ | ε-Greedy 정책 |
| 6-1 | $Q_n = Q_{n-1} + \alpha(G^{(n)} - Q_{n-1})$ | Exponential Moving Average 업데이트 |

---

## 핵심 기호 사전

| 기호 | 이름 | 의미 |
|------|------|------|
| $G^{(n)}$ | n번째 에피소드 Return | n번째 에피소드에서 상태 $s$(또는 $(s,a)$)를 방문한 후 얻은 실제 누적 보상 |
| $\varepsilon$ | 탐색률 (epsilon) | ε-Greedy에서 무작위 탐색 확률 |
| $\alpha$ | 학습률 (learning rate) | 고정 스텝 사이즈, 최신 데이터에 부여하는 가중치 |
| $n_s$ | 방문 횟수 | 상태 $s$를 방문한 누적 횟수 |
| $Q(s, a)$ | Q함수 / 행동 가치 함수 | 상태 $s$에서 행동 $a$를 취했을 때 기대 누적 보상 |
| reversed(memory) | 역방향 순회 | 에피소드 데이터를 끝부터 역순으로 처리 |
| Exploration | 탐색 | 새로운 행동을 시도하여 환경 정보 수집 |
| Exploitation | 활용 | 현재까지 학습된 정보를 활용하여 최선 행동 선택 |

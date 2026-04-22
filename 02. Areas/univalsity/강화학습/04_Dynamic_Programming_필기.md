---
title: "Dynamic Programming"
subtitle: "DP를 이용한 Policy Evaluation & Improvement"
subject: "지능 시스템 / 강화학습"
professor: "박태형 (충북대학교 지능시스템·로보틱스학과)"
date_created: 2026-03-25
source_file: "04-Dynamic Programming.pdf"
total_pages: 43
tags:
  - DynamicProgramming
  - ReinforcementLearning
  - PolicyEvaluation
  - PolicyIteration
  - ValueIteration
  - BellmanEquation
  - MDP
  - GridWorld
status: 학습중
---

# Dynamic Programming

> **출처**: 04-Dynamic Programming.pdf | Prof. Tae-Hyoung Park, Dept. of Intelligent Systems & Robotics, CBNU
> **총 43페이지** — 각 섹션 제목 옆에 `[p.XX]` 형식으로 원본 슬라이드 번호를 표기함

---

## 목차

- [[#1. MDP 요약]]
- [[#2. Dynamic Programming 개요]]
- [[#3. Iterative Bellman Equation]]
- [[#4. Policy Evaluation (정책 평가)]]
- [[#5. GridWorld Class 구현]]
- [[#6. Bellman Optimality Equation]]
- [[#7. Policy Improvement (정책 개선)]]
- [[#8. Policy Iteration Method (정책 반복법)]]
- [[#9. Value Iteration Method (가치 반복법)]]
- [[#10. Policy Iteration vs. Value Iteration 비교]]
- [[#11. 요약 및 한계점]]
- [[#🧠 내 생각 / 의문점]]
- [[#📝 시험 대비 핵심 정리]]
- [[#❓ 퀴즈]]

---

## 1. MDP 요약

> 📄 **원본 슬라이드**: p.2

> [!tip] MDP란?
> **Markov Decision Process**의 약자. 에이전트(Agent)가 환경(Environment)과 상호작용하며 보상을 최대화하는 결정을 내리는 수학적 틀이다. 상태(State), 행동(Action), 보상(Reward), 정책(Policy)으로 구성된다.

### 핵심 구성 요소

| 구성 요소 | 기호 | 설명 |
|-----------|------|------|
| State (상태) | $S_t$ | 현재 환경의 상태 |
| Action (행동) | $A_t$ | 에이전트가 선택하는 행동 |
| Reward (보상) | $R_t$ | 행동 후 받는 즉각적인 보상 |
| Policy (정책) | $\pi(a\|s)$ | 상태 s에서 행동 a를 선택할 확률 |
| Return (반환) | $G_t$ | 미래 보상의 누적합 (할인 적용) |

$$G_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \cdots$$

### Value 함수

- **State-value**: $v_\pi(s) = \mathbb{E}_\pi[G_t \| S_t = s]$
- **Action-value**: $q_\pi(s, a) = \mathbb{E}_\pi[G_t \| S_t = s, A_t = a]$

**Bellman Equation**:

$$v_\pi(s) = \sum_{a,s'} \pi(a|s) \, p(s'|s,a) \{r(s,a,s') + \gamma v_\pi(s')\}$$

$$q_\pi(s,a) = \sum_{s'} p(s'|s,a) \left\{r(s,a,s') + \gamma \sum_{a'} \pi(a'|s') q_\pi(s',a')\right\}$$

### Optimal Value

- **최적 상태 가치**: $v_*(s) = \max_\pi v_\pi(s)$
- **최적 행동 가치**: $q_*(s,a) = \max_\pi q_\pi(s,a)$

**Bellman Optimality Equation**:

$$v_*(s) = \max_a \sum_{s'} p(s'|s,a)\{r(s,a,s') + \gamma v_*(s')\}$$

**최적 정책 (결정론적)**:

$$\pi_*(a|s) = \begin{cases} 1, & \text{if } a = \arg\max_{a \in \mathcal{A}} q_*(s,a) \\ 0, & \text{otherwise} \end{cases}$$

$$\mu_*(s) = \arg\max_a q_*(s,a) \quad \text{(optimal action)}$$

---

## 2. Dynamic Programming 개요

> 📄 **원본 슬라이드**: p.3–5

### Policy Evaluation & Improvement의 흐름 [p.3]

| 단계 | 방법 | 목표 |
|------|------|------|
| **Policy Evaluation (평가)** | Bellman Equation | $q_\pi(s,a)$, $v_\pi(s)$ 계산 |
| **Policy Improvement (개선)** | Bellman Optimality Equation | $q_*(s,a)$, $\pi_*(a\|s)$ 계산 |

```
π(a|s) ──평가──→ v_π(s), q_π(s,a)
   ↑                        │
   └──────개선───────────────┘
π*(a|s), q*(s,a)
```

> [!info] "How to solve ?" 핵심 질문
> 연립방정식 형태인 Bellman Equation을 **어떻게 효율적으로 푸는가?**가 이 강의의 핵심 문제다. 이 해답이 바로 Dynamic Programming이다.ov Decision Processes (MDPs Equation, which is crucial for solving Mark improvement in reinforcement learning contexts. It explores foundational concepts like the Bellman dynamic programming techniques can be applied for policy evaluation and of how an in-depth understanding lecture or material is to provide primary objective of this):

The적** (Objective

### DP의 핵심 아이디어 [p.4]

> [!tip] Dynamic Programming이란?
> 복잡한 문제를 **작은 부분 문제(subproblem)로 분해**하고, 그 결과를 **저장**해서 재계산을 피하는 알고리즘 전략이다. 강화학습에서는 Bellman Equation을 반복적으로 풀어 가치 함수를 계산하는 데 활용한다.

두 가지 핵심 특성:
1. **분해**: 복잡한 문제 → 더 단순한 부분 문제들로 분해
2. **저장(메모이제이션)**: 부분 문제의 결과를 저장 → 중복 계산 방지

> [!warning] Curse of Dimensionality (차원의 저주)
> 상태 공간의 차원이 증가할수록 계산량이 **지수적으로 폭발**한다. DP의 가장 큰 한계점으로, 복잡한 실세계 문제에 적용하기 어렵다.

### 예제 1: Assembly Scheduling Problem [p.4]

조립 라인 스케줄링 문제의 점화식:

$$f_1[1] = e_1 + a_{1,1}, \quad f_2[1] = e_2 + a_{2,1}$$

$$f_1[j] = \min(f_1[j-1] + a_{1,j},\ f_2[j-1] + t_{2,1} + a_{1,j})$$
$$f_2[j] = \min(f_2[j-1] + a_{2,j},\ f_1[j-1] + t_{1,2} + a_{2,j})

The document titled "Dynamic Programming" with the subtitle "DP를 이용한 Policy Evaluation & Improvement" is a lecture or study material related to intelligent systems and reinforcement learning. It was created on March 25, 2026, by Professor 박태형 from the 충북대학교 지능시스템·로보틱스학과. The document contains a total of 43 pages and covers topics such as Dynamic Programming, Policy Evaluation, Policy Iteration, Value Iteration, Bellman Equation, Markov Decision Processes (MDP), and GridWorld.

**목적** (Objective):

The primary objective of this lecture or material is to provide an in-depth understanding of how dynamic programming techniques can be applied for policy evaluation and improvement in reinforcement learning contexts. It explores foundational concepts like the Bellman Equation, which is crucial for solving Markov Decision Processes (MDPs), and explains how these methods can optimize decision-making processes within specified environments such as GridWorld.

This content is likely part of a broader curriculum focused on enhancing knowledge in intelligent systems and robotics through reinforcement learning strategies. The use of dynamic programming serves as a key technique for improving policy decisions systematically through iterative approaches like policy iteration and value iteration. like policy iteration and policy decisions systematically through iterative approaches$$

$$f^* = \min(f_1[n] + x_1,\ f_2[n] + x_2)$$

> [!info] 왜 이 예제를 드는가?
> 각 스테이션에서의 최솟값을 이전 단계의 결과로부터 점화식으로 계산하는 것이 DP의 전형적인 예다. 이전 결과를 저장해 두고 다음 단계 계산에 재사용한다.

### 예제 2: Longest Common Subsequence (LCS) [p.5]

$$c[i,j] = \begin{cases} 0 & i=0 \text{ or } j=0 \\ c[i-1,j-1]+1 & i,j>0 \text{ and } x_i = y_j \\ \max(c[i,j-1],\ c[i-1,j]) & i,j>0 \text{ and } x_i \neq y_j \end{cases}$$

> [!tip] LCS란?
> 두 수열에서 공통으로 나타나는 가장 긴 부분 수열을 찾는 문제. 예: "ABCBDAB"와 "BDCAB"의 LCS는 "BCAB" (길이 4). 이전 계산 결과를 2D 테이블에 저장하며 풀어나간다.

---

## 3. Iterative Bellman Equation

> 📄 **원본 슬라이드**: p.6–7

### 기본 Bellman Equation의 문제점 [p.6]

$$v_\pi(s) \equiv \mathbb{E}_\pi[G_t|S_t=s] = \mathbb{E}_\pi[R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \cdots | S_t = s]$$

$$v_\pi(s) = \sum_{a,s'} \pi(a|s)\, p(s'|s,a)\{r(s,a,s') + \gamma v_\pi(s')\}$$

- '다음 상태'의 가치함수 $v_\pi(s')$ 와 '현재 상태'의 가치함수 $v_\pi(s)$ 의 관계식
- **연립방정식 문제** → 복잡한 문제에 직접 적용하기 어려움

### Iterative Bellman Equation (핵심!) [p.6]

$$\boxed{V_{k+1}(s) = \sum_{a,s'} \pi(a|s)\, p(s'|s,a)\{r(s,a,s') + \gamma V_k(s')\}}$$

$$V_0(s) \to V_1(s) \to \cdots \to V_k(s) \to V_{k+1}(s) \to \cdots \approx v_\pi(s) \quad \text{(dynamic programming)}$$

> [!tip] Bootstrapping이란?
> 자신의 추정치로 다음 추정치를 개선하는 방법. $V_k(s')$ (이전 추정)으로부터 $V_{k+1}(s)$ (새 추정)를 계산한다. "자기 자신의 부츠끈을 잡아 올린다"는 뜻에서 유래.

핵심 포인트:
- $V_k(s)$: k번째로 갱신된 state-value의 **'추정치'** (실제 $v_\pi(s)$와 다름!!)
- 추정치 $V_k(s')$로부터 추정치 $V_{k+1}(s)$를 개선 → **bootstrapping**
- 연립방정식 문제를 **dynamic programming으로 전환**하여 해결

### Deterministic State-Transition Case [p.7]

결정론적 상태 전이 확률:

$$p(s'|s,a) = \begin{cases} 1, & \text{if } s' = f(s,a) \\ 0, & \text{if } s' \neq f(s,a) \end{cases}$$

확률적 → 결정론적 단순화:

$$V_{k+1}(s) = \sum_{a,s'} \pi(a|s)\, p(s'|s,a)\{r(s,a,s') + \gamma V_k(s')\}$$

$$\to \quad V_{k+1}(s) = \sum_a \pi(a|s) \sum_{s'} p(s'|s,a)\{r(s,a,s') + \gamma V_k(s')\}$$

$$\to \quad \boxed{V_{k+1}(s) = \sum_a \pi(a|s)\{r(s,a,s') + \gamma V_k(s')\}} \quad s' = f(s,a)$$

> [!info] 왜 결정론적 케이스가 중요한가?
> GridWorld 같은 결정론적 환경에서는 전이 확률의 합 계산이 불필요해져 계산이 단순해진다. 실습에서 구현하는 GridWorld가 이 케이스에 해당한다.

---

## 4. Policy Evaluation (정책 평가)

> 📄 **원본 슬라이드**: p.8–23

### 목적 상태에 대한 V(s) 갱신량의 최댓값. Δ가 임계값 θ보다 작아지면 수렴했다고 판단하고 반복을 멈춘다.

반복에 따른 수렴 과정 (4×4 Grid 예시):

It seems like you're referring to a document or notes titled "04_Dynamic_Programming_필기" and mentioning a specific example involving a "4×4 Grid." Dynamic programming is a

| k | 특징 |
|---|------|
| k=0 | 모두 0으로 초기화 |
| k=1 | 주변 셀들이 -1로 갱신 |
| k=2 | 더 큰 음수 값으로 확산 |
| k=10 | 점차 안정적인 형태 |
| k=∞ | 진짜 $v_\pi(s)$ 수렴 |

### 2-Grid World 예제 [p.9]

환경 설정:
- **L1 → L2 이동**: 보상 +1
- **벽에 부딪힘**: 보상 −1
- **정책 π**: 0.5(LEFT) + 0.5(RIGHT) 균등 정책
- **할인율 γ = 0.9**

$V_0$ 초기화: $V_0(L1) = 0, \quad V_0(L2) = 0$

$V_1$ 계산:

$$V_1(L1) = \pi(LEFT|L1)\{r(L1,LEFT,L1) + \gamma V_0(L1)\} + \pi(RIGHT|L1)\{r(L1,RIGHT,L2) + \gamma V_0(L2)\}$$
$$= 0.5\{-1 + 0.9 \times 0\} + 0.5\{1 + 0.9 \times 0\} = 0$$

$$V_1(L2) = 0.5\{0 + 0.9 \times 0\} + 0.5\{-1 + 0.9 \times 0\} = -0.5$$

$$V_0(s): [0\ \ 0] \quad \to \quad V_1(s): [0\ \ {-0.5}]$$

### 구현 방법 비교 [p.10–11]

#### 방법 1: Copy 방법 [p.10]

```python
V = {'L1': 0.0, 'L2': 0.0}
new_V = V.copy()  # Dictionary data structure

cnt = 0
while True:
    new_V['L1'] = 0.5 * (-1 + 0.9 * V['L1']) + 0.5 * (1 + 0.9 * V['L2'])
    new_V['L2'] = 0.5 * (0 + 0.9 * V['L1']) + 0.5 * (-1 + 0.9 * V['L2'])

    delta = abs(new_V['L1'] - V['L1'])
    delta = max(delta, abs(new_V['L2'] - V['L2']))

    V = new_V.copy()
    cnt += 1
    if delta < 0.0001:
        print(V)
        print('갱신 횟수:', cnt)
        break
```

결과: $V_{76}(L1) = -2.249,\quad V_{76}(L2) = -2.749$ (76회 갱신)

#### 방법 2: In-place 방법 [p.11]

```python
V = {'L1': 0.0, 'L2': 0.0}

cnt = 0
while True:
    t = 0.5 * (-1 + 0.9 * V['L1']) + 0.5 * (1 + 0.9 * V['L2'])
    delta = abs(t - V['L1'])
    V['L1'] = t  # 즉시 갱신 (copy 없이)

    t = 0.5 * (0 + 0.9 * V['L1']) + 0.5 * (-1 + 0.9 * V['L2'])
    delta = max(delta, abs(t - V['L2']))
    V['L2'] = t

    cnt += 1
    if delta < 0.0001:
        print(V)
        print('갱신 횟수:', cnt)
        break
```

결과: $V_{60}(L1) = -2.249,\quad V_{60}(L2) = -2.749$ (**60회**로 더 빠름)

> [!tip] In-place vs Copy 방법
> - **Copy 방법**: 모든 상태를 동시에 갱신 (배치 갱신). V와 new_V 두 배열 필요
> - **In-place 방법**: 갱신 즉시 사용. 메모리 효율적이고 **수렴이 더 빠름** (76 → 60회)


한국어로 다음을 설명해 In-place vs Copy 방법 에 대해 더 자세하게 설명해줘 짱짱쉽게

 In-place와 Copy 방법은 데이터를 처리하거나 알고리즘을 구현할 때 사용하는 두 가지 접근 방식입니다. 이 두 방법의 차이점과 장단점을 짱짱 쉽게 설명해드릴게요.

### In-place 방법

**In-place**는 말 그대로 "제자리에서" 작업을 수행하는 것을 의미합니다. 즉, 데이터를 저장하고 있는 기존 메모리를 그대로 사용하면서 변경을 가하는 방식입니다. 새로운 메모리를 할당하지 않기 때문에 공간 효율성이 좋습니다.

- **장점**:
  - 메모리 사용량이 적습니다. 새로운 공간을 마련할 필요가 없으니까요.
  - 빠르게 실행될 수 있습니다. 추가적인 메모리 할당이나 복사가 없기 때문입니다.

- **단점**:
  - 원래 데이터를 변경하기 때문에, 작업 후에 원본 데이터가 필요하다면 문제가 됩니다.
  - 실수로 데이터를 변경하면 복구가 어렵습니다.

### Copy 방법

**Copy** 방법은 현재 데이터의 사본을 만들어서 그 사본에 작업을 수행하는 방식입니다. 원본 데이터는 그대로 유지되고, 변경된 결과는 새로 생성된 사본에 반영됩니다.

- **장점**:
  - 원본 데이터가 손상되지 않으므로 안전하게 사용할 수 있습니다.
  - 여러 버전의 데이터를 유지해야 할 때 유용합니다.

- **단점**:
  - 더 많은 메모리를 사용합니다. 특히 대량의 데이터를 다룰 때 비효율적일 수 있습니다.
  - 메모리를 추가로 할당하고 복사하는 과정에서 시간이 더 걸릴 수 있습니다.

### 예시로 이해하기

1. **In-place 예시**: 방 안에 있는 가구의 배치를 바꾸는 것과 같습니다. 방 자체를 바꾸지 않고 내부의 물건 위치만 조정합니다.

2. **Copy 예시**: 방 안에 있는 모든 물건들을 다른 빈 방으로 옮겨서 배치를 바꾼 후, 새로운 방에서 어떻게 변화되었는지 확인하는 것과 같습니다.

이 두 가지 방법은 각자의 상황에 맞게 선택해서 사용하면 됩니다. 만약 메모리가 제한적이고 속도가 중요하다면 in-place를, 데이터의 안전성과 변형 전후 비교가 중요하다면 copy를 사용하는 것이 좋습니다.

### Policy Evaluation 요약 흐름 [p.23]

```
전통적 방법:
p(s'|s,a), r(s,a,s'), π(a|s) → 벨만 방정식 → 연립방정식 계산기 → v_π(s)

↓ Dynamic Programming으로 대체

p(s'|s,a), r(s,a,s'), π(a|s) → Dynamic Programming → v_π(s)
```

---

## 5. GridWorld Class 구현

> 📄 **원본 슬라이드**: p.12–22

### 3×4 Grid World 환경 설정 [p.12]

| 항목 | 내용 |
|------|------|
| 에이전트 이동 | 상하좌우 4방향 |
| 회색 칸 | 벽 (들어갈 수 없음) |
| 그리드 경계 | 벽으로 둘러싸임 (나갈 수 없음) |
| 벽 충돌 보상 | 0 |
| 사과 보상 | +1 (목표 상태, 종료) |
| 폭탄 보상 | −1 |
| 상태 전이 | 결정론적 (deterministic) |
| 에피소드 | 일회성 (사과 획득 시 종료) |

### GridWorld 클래스 - 초기화 [p.13]

```python
import numpy as np

class GridWorld:
    def __init__(self):
        self.action_space = [0, 1, 2, 3]  # 행동 공간(가능한 행동들)
        self.action_meaning = {            # 행동의 의미
            0: 'UP',
            1: 'DOWN',
            2: 'LEFT',
            3: 'RIGHT',
        }

        self.reward_map = np.array(        # 보상 맵(각 좌표의 보상값)
            [[0, 0, 0, 1.0],
             [0, None, 0, -1.0],
             [0, 0, 0, 0]]
        )
        self.goal_state = (0, 3)   # 목표 상태(좌표)
        self.wall_state = (1, 1)   # 벽 상태(좌표)
        self.start_state = (2, 0)  # 시작 상태(좌표)
        self.agent_state = self.start_state  # 에이전트 초기 상태(좌표)
```

> [!info] 좌표 체계
> reward_map의 인덱스: 첫 번째 축(행)이 세로 방향, 두 번째 축(열)이 가로 방향. goal_state = (0, 3)은 0행 3열 = 오른쪽 상단.

### GridWorld 클래스 - 주요 메서드 [p.14–16]

```python
@property
def height(self): return len(self.reward_map)
@property
def width(self): return len(self.reward_map[0])
@property
def shape(self): return self.reward_map.shape

def actions(self): return self.action_space  # [0, 1, 2, 3]

def states(self):
    for h in range(self.height):
        for w in range(self.width):
            yield (h, w)
```

**next_state()**: 다음 상태 계산 = $s' = f(s,a)$ [p.15]

```python
def next_state(self, state, action):
    action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # UP/DOWN/LEFT/RIGHT
    move = action_move_map[action]
    next_state = (state[0] + move[0], state[1] + move[1])
    ny, nx = next_state

    # 경계 밖이거나 벽이면 현재 상태 유지
    if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
        next_state = state
    elif next_state == self.wall_state:
        next_state = state

    return next_state

def reward(self, state, action, next_state):  # r(s,a,s')
    return self.reward_map[next_state]
```

**render_v()**: 가치함수 시각화 [p.16]

```python
env = GridWorld()
V = {}
for state in env.states():
    V[state] = np.random.randn()  # 더미 상태 가치함수
env.render_v(V)
```

### defaultdict 활용 [p.17]

> [!tip] defaultdict란?
> 파이썬 `collections` 모듈의 딕셔너리 자료구조. 존재하지 않는 키에 접근해도 **자동으로 기본값을 생성**한다. 명시적 초기화 루프 없이 간결하게 V(s)와 π(a|s)를 표현할 수 있다.

```python
from collections import defaultdict

# V_k(s) 초기화 - 모든 상태의 가치를 0으로
V = defaultdict(lambda: 0)

# π(a|s) 초기화 - 균등 정책 (각 행동 0.25 확률)
pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
```

기존 방법(명시적 초기화):
```python
V = {}
for state in env.states():
    V[state] = 0
```

### One-Step Evaluation: $V_k \to V_{k+1}$ [p.18]

```python
def eval_onestep(pi, V, env, gamma=0.9):
    for state in env.states():
        if state == env.goal_state:  # 목표 상태의 가치 함수는 항상 0
            V[state] = 0
            continue

        action_probs = pi[state]  # probs = probabilities(확률)의 약자
        new_V = 0

        for action, action_prob in action_probs.items():
            # a: π(a|s) = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            # V_{k+1}(s) = Σ π(a|s){r(s,a,s') + γV_k(s')}
            new_V += action_prob * (r + gamma * V[next_state])

        V[state] = new_V
    return V
```

### Iterative Evaluation: $V_0 \to V_1 \to \cdots \approx v_\pi$ [p.19]

```python
def policy_eval(pi, V, env, gamma, threshold=0.001):
    while True:
        old_V = V.copy()         # 갱신 전 가치 함수
        V = eval_onestep(pi, V, env, gamma)  # V_k → V_{k+1}

        # 갱신된 양의 최댓값 계산
        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t        # Δ ← max(Δ, |v − V(s)|)

        if delta < threshold:    # ≈ v_π(s)
            break
    return V
```

threshold  임계값, : 편차량이 0.001 이하가 되면 종료를 하겠다. 

### Policy Evaluation 결과 (3×4 Grid World) [p.20, 22]

```python
env = GridWorld()
gamma = 0.9  # 할인율
pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})  # 무작위 정책
V = defaultdict(lambda: 0)  # 가치 함수
V = policy_eval(pi, V, env, gamma)  # 정책 평가
env.render_v(V, pi)  # 시각화
```

결과: 사과(+1) 방향으로 갈수록 양수, 폭탄(-1) 근처는 음수 값을 보임

---

## 6. Bellman Optimality Equation

> 📄 **원본 슬라이드**: p.24

기본 Bellman Equation (정책 π에 대한):
$$v_\pi(s) = \sum_{a,s'} \pi(a|s)\, p(s'|s,a)\{r(s,a,s') + \gamma v_\pi(s')\}$$
$$q_\pi(s,a) = \sum_{s'} p(s'|s,a)\left\{r(s,a,s') + \gamma \sum_{a'} \pi(a'|s') q_\pi(s',a')\right\}$$

**Bellman Optimality Equation** (최적 정책에 대한):
$$v_*(s) = \max_a \sum_{s'} p(s'|s,a)\{r(s,a,s') + \gamma v_*(s')\}$$
$$q_*(s,a) = \sum_{s'} p(s'|s,a)\left\{r(s,a,s') + \gamma \max_{a'} q_*(s',a')\right\}$$

### 최적 정책 유도

| 정책 유형                    | 수식                                                        | 예시                                           |
| ------------------------ | --------------------------------------------------------- | -------------------------------------------- |
| **Stochastic policy**    | $\pi_*(a\|s) > 0$ only where $q_*(s,a) = \max_b q_*(s,b)$ | [0.25, 0.25, 0.25, 0.25] 또는 [0.5, 0, 0.5, 0] |
| **Deterministic policy** | $\pi_*(a\|s) = 1$ if $a = \mu_*(s)$, else 0               | [1,0,0,0] 또는 [0,0,1,0]                       |


해당  개념에 대해 한글로 쉽게 설명해봐  **Stochastic policy**   vs **Deterministic policy**

**Stochastic Policy**와 **Deterministic Policy**는 주로 강화 학습에서 사용되는 개념으로, 에이전트가 환경 내에서 행동을 결정하는 방법을 설명합니다. 이 두 가지 정책의 차이를 쉽게 설명해볼게요.

1. **Stochastic Policy (확률적 정책):**
   - 확률적 정책은 에이전트가 어떤 상태에 있을 때 여러 가능한 행동 중 하나를 확률적으로 선택하는 방식입니다.
   - 즉, 특정 상태에서 여러 행동에 대한 확률 분포가 존재하며, 이 확률에 따라 행동이 무작위로 결정됩니다.
   - <mark style="background: #FFB86CA6;">예를 들어, 어떤 상태에서 행동 A를 선택할 확률이 70%이고, 행동 B를 선택할 확률이 30%라면, 이 상태에서는 대체로 A를 선택하지만 가끔씩 B도 선택할 수 있습니다.</mark>
   - 확률적 정책은 다양한 상황에 적응하고 탐색을 계속할 수 있는 유연성을 제공합니다.

2. **Deterministic Policy (결정적 정책):**
   - 결정적 정책은 특정 상태에서 항상 같은 행동을 선택하는 방식입니다.
   - 즉, 각 상태마다 하나의 최선의 행동이 고정되어 있으며, 그 상태에서는 항상 그 동일한 행동을 수행합니다.
   - 예를 들어, 어떤 상태에서는 항상 행동 A만을 선택하게 됩니다.
   - 결정적 정책은 단순하고 예측 가능하지만, 환경 변화나 불확실성에 적응하기 어려울 수 있습니다.

두 정책 모두 장단점이 있으며, 문제의 성격이나 학습 환경에 따라 적절한 방식을 선택하여 사용할 수 있습니다. Stochastic policy는 특히 초기 탐색 단계나 불확실성이 큰 환경에서 유리하며, Deterministic policy는 명확한 최적의 전략이 있을 때 사용하기 좋습니다.

최적 행동:
$$\mu_*(s) = \arg\max_a q_*(s,a) = \arg\max_a \{r(s,a,s') + \gamma v_*(s')\} \quad \text{(deterministic case)}$$

---

## 7. Policy Improvement (정책 개선)

> 📄 **원본 슬라이드**: p.25

### Policy Improvement Theorem

> Let π and π' be any pair of **deterministic policies** such that, for all $s \in \mathcal{S}$:
> $$q_\pi(s, \mu'(s)) \geq v_\pi(s)$$
> Then: $v_{\pi'}(s) \geq v_\pi(s)$
> 즉, **정책 π'는 π보다 같거나 더 좋다**.

여기서 $\pi'(a|s) = 1$ if $a = \mu'(s)$, else $0$

> [!info] 직관적 이해
> 현재 정책 π보다 각 상태에서 더 높은 action-value를 주는 행동 μ'(s)을 선택하면, 새로운 정책 π'는 반드시 π보다 같거나 좋다. 이것이 greedy 개선의 보장이다.

### 최적 정책으로의 수렴 과정

$$\pi_0 \xrightarrow{E} v_{\pi_0} \xrightarrow{I} \pi_1 \xrightarrow{E} v_{\pi_1} \xrightarrow{I} \pi_2 \xrightarrow{E} \cdots \xrightarrow{I} \pi_* \xrightarrow{E} v_*$$

- **E**: Policy Evaluation (평가)
- **I**: Policy Improvement (개선)

두 가지 반복법:
1. **Policy Iteration Method** (정책 반복법)
2. **Value Iteration Method** (가치 반복법)

---

## 8. Policy Iteration Method (정책 반복법)

> 📄 **원본 슬라이드**: p.26–33

### 알고리즘 개요 [p.26]

**Policy Iteration** = **Iterative Policy Evaluation** + **Policy Improvement (greedy)**

```
1. Initialization
   V(s) ∈ ℝ and π(s) ∈ A(s) arbitrarily for all s ∈ S

2. Policy Evaluation
   Repeat
       Δ ← 0
       For each s ∈ S:
           v ← V(s)
           V(s) ← Σ_{s',r} p(s',r|s,π(s))[r + γV(s')]
           Δ ← max(Δ, |v − V(s)|)
   until Δ < θ

3. Policy Improvement
   policy-stable ← true
   For each s ∈ S:
       a ← π(s)
       π(s) ← argmax_a Σ_{s',r} p(s',r|s,a)[r + γV(s')]
       If a ≠ π(s), then policy-stable ← false
   If policy-stable, then stop and return V and π; else go to 2
```

- **Evaluation**: Bellman equation → $v_\pi$ 계산
- **Improvement**: greedy selection → 새로운 $\pi$ 선택

$$\mu_*(s) = \arg\max_a \sum_{s'} p(s'|s,a)\{r(s,a,s') + \gamma v_\pi(s')\}$$

### argmax() 구현 [p.28]

```python
def argmax(d):
    max_value = max(d.values())
    max_key = 0
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key

# 예시: 9.9가 최댓값 (키는 2)
action_values = {0: 0.1, 1: -0.3, 2: 9.9, 3: -1.3}
max_action = argmax(action_values)
print(max_action)  # 출력: 2
```

### greedy_policy() 구현 [p.29]

```python
def greedy_policy(V, env, gamma):
    pi = {}
    for state in env.states():
        action_values = {}

        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]          # ① q(s,a) = r(s,a,s') + γv_π(s)
            action_values[action] = value

        max_action = argmax(action_values)             # ② μ*(s) = argmax_a q(s,a)
        action_probs = {0: 0, 1: 0, 2: 0, 3: 0}
        action_probs[max_action] = 1.0
        pi[state] = action_probs                       # ③ π*(a|s)
    return pi
```

### policy_iter() 구현 [p.30]

```python
def policy_iter(env, gamma, threshold=0.001, is_render=False):
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0)

    while True:
        V = policy_eval(pi, V, env, gamma, threshold)  # ① 평가 (Policy evaluation)
        new_pi = greedy_policy(V, env, gamma)           # ② 개선 (Policy improvement)

        if is_render:
            env.render_v(V, pi)

        if new_pi == pi:  # ③ 갱신 여부 확인 → 변화 없으면 수렴
            break
        pi = new_pi

    return pi  # → π*(a|s)
```

### 실행 결과 [p.31, 33]

```python
env = GridWorld()
gamma = 0.9
pi = policy_iter(env, gamma)
```

초기 (무작위 정책) → 최적 정책 (4번째 갱신 후):
- 초기: 모든 방향으로 균등하게 분산된 화살표, 음수 가치
- 최종: 사과를 향하는 명확한 방향 화살표, 양수 가치 (0.81~1.00 범위)

---

## 9. Value Iteration Method (가치 반복법)

> 📄 **원본 슬라이드**: p.34–41

### Policy Iteration vs Value Iteration 비교 [p.34–35]

| 구분 | Policy Iteration | Value Iteration |
|------|-----------------|-----------------|
| **평가 방정식** | Bellman Equation | Bellman **Optimality** Equation |
| **계산식** | $V'(s) = \sum_{a,s'} \pi(a\|s)\, p(s'\|s,a)\{r+\gamma V(s')\}$ | $V'(s) = \max_a \sum_{s'} p(s'\|s,a)\{r+\gamma V(s')\}$ |
| **정책 π 필요 여부** | O (계산 시 π(a\|s) 또는 μ(s) 필요) | **X** (불필요!) |
| **특징** | Eval(수렴) + Improve 반복 | Eval+Improve 동시 수행 |
| **수렴 속도** | 상대적으로 느림 | **일반적으로 더 빠름** |
| **불필요한 계산** | 있음 | 제거됨 |

**개선 방정식** (동일):
$$\mu(s) = \arg\max_a \sum_{s'} p(s'|s,a)\{r(s,a,s') + \gamma V(s')\}$$

> [!info] Value Iteration의 핵심 차이
> Policy Iteration은 먼저 현재 정책으로 완전히 수렴할 때까지 평가하고 그다음에 개선한다. Value Iteration은 **매 스텝마다 평가와 개선을 동시에** 수행한다. max 연산 자체가 greedy improvement를 내포하고 있기 때문이다.

### value_iter_onestep() [p.36]

```python
def value_iter_onestep(V, env, gamma):
    for state in env.states():           # ① 모든 상태에 차례로 접근
        if state == env.goal_state:      # ② 목표 상태의 가치 함수는 항상 0
            V[state] = 0
            continue

        action_values = []
        for action in env.actions():     # ③ 모든 행동에 차례로 접근
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            # s' = f(s,a),  r(s,a,s') + γV(s')
            value = r + gamma * V[next_state]   # ③ 새로운 가치 함수
            action_values.append(value)

        V[state] = max(action_values)    # ④ 최댓값 추출
        # V'(s) = max_a {r(s,a,s') + γV'(s')}
    return V
```

### value_iter() [p.37]

```python
def value_iter(V, env, gamma, threshold=0.001, is_render=True):
    while True:
        if is_render:
            env.render_v(V)

        old_V = V.copy()        # 갱신 전 가치 함수
        V = value_iter_onestep(V, env, gamma)   # V'_k(s), k = 1, 2, ...

        # 갱신된 양의 최댓값 구하기
        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t

        if delta < threshold:
            break
    return V
```

### Optimal Policy 도출 [p.38]

```python
V = defaultdict(lambda: 0)
env = GridWorld()
gamma = 0.9

V = value_iter(V, env, gamma)           # 최적 가치 함수 찾기
pi = greedy_policy(V, env, gamma)       # 최적 정책 찾기
env.render_v(V, pi)
```

초기(모두 0) → 마지막(세 번째 갱신) → 최종 최적 정책:
- 0.00 → 수렴 과정 → 0.81~1.00 양수 가치 + 방향 화살표

---

## 10. Policy Iteration vs. Value Iteration 비교

> 📄 **원본 슬라이드**: p.39

| | Policy Iteration | Value Iteration |
|--|-----------------|-----------------|
| **코드 핵심** | `V = policy_eval(...)` + `new_pi = greedy_policy(...)` | `V = value_iter(...)` + `pi = greedy_policy(...)` |
| **수렴까지 갱신** | 4회 (정책 갱신 기준) | 3회 (가치 함수 수렴 기준) |
| **장점** | 직관적, 정책 명시적 관리 | 빠른 수렴, 정책 불필요 |

> [!warning] 두 방법의 결과는 동일!
> 두 방법 모두 같은 최적 정책 $\pi_*$와 최적 가치 함수 $v_*$로 수렴한다. 차이는 **효율성**에 있다. GridWorld처럼 단순한 환경에서는 차이가 작지만, 복잡한 환경에서는 Value Iteration이 유리하다.

---

## 11. 요약 및 한계점

> 📄 **원본 슬라이드**: p.43

### Optimal Policy를 어떻게 구하는가?

**Policy Improvement Procedure**:

$$\pi_0 \xrightarrow{E} v_{\pi_0} \xrightarrow{I} \pi_1 \xrightarrow{E} v_{\pi_1} \xrightarrow{I} \pi_2 \xrightarrow{E} \cdots \xrightarrow{I} \pi_* \xrightarrow{E} v_*$$

| 방법 | 평가 | 개선 |
|------|------|------|
| **Policy Iteration** | Bellman Equation | Greedy Selection (분리 수행) |
| **Value Iteration** | Bellman Optimality Equation | Greedy Selection (동시 수행) |

- **Policy Evaluation** → **Dynamic Programming**으로 해결
- **Policy Improvement** → **Greedy Selection**으로 해결

### Dynamic Programming의 한계점

> [!warning] DP의 두 가지 핵심 한계
> 1. **Curse of Dimensionality (차원의 저주)**: 상태 공간이 커질수록 계산량이 폭발적으로 증가
> 2. **환경 모델 필요**: $p(s'|s,a)$, $r(s,a,s')$를 알아야 함 → 실제 환경에서는 모르는 경우가 많음

이 두 한계를 극복하기 위해 다음 단계에서는 **Model-free** 방법(Q-learning, SARSA 등)을 배우게 된다.

---

## 실습 정리

> 📄 **원본 슬라이드**: p.21–22, 32–33, 40–41

| 실습 | 파일 | 내용 |
|------|------|------|
| **실습 #1** [p.21–22] | `gridworld.py`, `policy_eval.py` | GridWorld 클래스 + Policy Evaluation 구현 |
| **실습 #2** [p.32–33] | `policy_iter.py` | Policy Iteration 구현 |
| **실습 #3** [p.40–41] | `value_iter.py` | Value Iteration 구현 |

### 실습 #3 핵심 구조 [p.40]

```python
from collections import defaultdict
from common.gridworld import GridWorld
from policy_iter import greedy_policy

def value_iter_onestep(V, env, gamma): ...
def value_iter(V, env, gamma, threshold=0.001, is_render=True): ...

if __name__ == '__main__':
    V = defaultdict(lambda: 0)
    env = GridWorld()
    gamma = 0.9

    V = value_iter(V, env, gamma)           # 최적 가치 함수 찾기
    pi = greedy_policy(V, env, gamma)       # 최적 정책 찾기
    env.render_v(V, pi)
```

---

## 퀴즈 문제 [p.42]

5×5 Grid World에 대한 value function 및 policy를 구하라:
- (1) Policy Iteration Method 사용
- (2) Value Iteration Method 사용

환경: 폭탄(-1) 2개, 사과(+1) 1개, 벽(회색) 2칸

---

## 🧠 내 생각 / 의문점

> *이 공간에 수업을 듣고 떠오르는 생각이나 의문점을 자유롭게 작성해 보세요*

### 💡 인사이트

-

### ❓ 의문점

- Policy Iteration과 Value Iteration 중 실제 프로젝트에서는 어떤 경우에 어떤 방법을 선택하는가?
- In-place 방법이 항상 더 빠르게 수렴하는가? 수렴 순서에 따라 달라지지 않는가?
- Curse of Dimensionality를 완화하기 위한 실질적인 방법에는 무엇이 있는가?

### 🔗 연관 개념

- 이전 주차 내용과 연결되는 부분: MDP (Markov Decision Process)의 수식적 토대 → DP로 해결
- 다음 단계: Model-free 방법 (Monte Carlo, TD Learning, Q-Learning 등)
- 실제 적용 가능한 프로젝트 아이디어: 미로 탈출 로봇, 게임 AI (체스, 바둑 등)

---

## 📝 시험 대비 핵심 정리

> **★★★ 반드시 암기할 내용**

### 1. Iterative Bellman Equation

$$V_{k+1}(s) = \sum_{a,s'} \pi(a|s)\, p(s'|s,a)\{r(s,a,s') + \gamma V_k(s')\}$$

- $V_k$: 실제 $v_\pi$가 아닌 **추정치**
- 반복을 통해 $v_\pi$에 수렴 (Bootstrapping)

### 2. Policy Evaluation 알고리즘

- 수렴 조건: $\Delta = \max_s |v - V(s)| < \theta$
- **Copy vs In-place**: In-place가 더 빠르게 수렴

### 3. Policy Iteration vs Value Iteration

| | Policy Iteration | Value Iteration |
|--|-----------------|-----------------|
| 평가 방정식 | Bellman Eq. (π 사용) | Bellman Optimality Eq. (max 사용) |
| π 필요 여부 | O | X |
| 수렴 속도 | 느림 | 빠름 |

### 4. Greedy Policy (최적 정책)

$$\mu_*(s) = \arg\max_a \{r(s,a,s') + \gamma V(s')\} \quad (s' = f(s,a))$$

### 5. DP의 한계

- **차원의 저주**: 계산량이 상태 차원에 지수적으로 증가
- **환경 모델 필요**: $p(s'|s,a)$, $r(s,a,s')$ 사전 지식 필요

---

## ❓ 퀴즈

**Q1.** Iterative Bellman Equation에서 $V_k(s)$와 실제 $v_\pi(s)$의 관계로 옳은 것은?

① $V_k(s) = v_\pi(s)$ (항상 같다)
② $V_k(s)$는 $v_\pi(s)$의 **추정치**이며, 반복이 증가할수록 수렴한다
③ $V_k(s)$는 $v_\pi(s)$보다 항상 크다
④ $V_k(s)$와 $v_\pi(s)$는 관계가 없다

> [!done]- 정답
> **②** $V_k(s)$는 추정치이고, $k \to \infty$일 때 $v_\pi(s)$로 수렴한다. Bootstrapping의 핵심 개념이다.

---

**Q2.** Policy Iteration과 Value Iteration의 핵심 차이점을 서술하라.

> [!done]- 정답
> - **Policy Iteration**: Bellman Equation 사용. 평가(수렴까지) 후 개선을 분리해서 수행. 계산 시 π(a|s) 필요.
> - **Value Iteration**: Bellman **Optimality** Equation 사용. max 연산으로 평가와 개선을 동시에 수행. π 불필요. 일반적으로 수렴이 더 빠르다.

---

**Q3.** In-place 방법이 Copy 방법보다 빠르게 수렴하는 이유는 무엇인가?

> [!done]- 정답
> In-place 방법은 V(s)를 갱신한 **직후 즉시** 다른 상태의 계산에 사용한다. Copy 방법은 이전 스텝의 값만 사용하므로 정보 전파가 느리다. In-place는 최신 정보를 더 빨리 활용하여 수렴이 빠르다.

---

**Q4.** Dynamic Programming이 실제 복잡한 환경에 적용하기 어려운 두 가지 이유를 설명하라.

① 차원의 저주(Curse of Dimensionality): 상태 공간이 커질수록 계산량이 **지수적으로 증가**한다.
② **환경 모델 필요**: 상태 전이 확률 $p(s'|s,a)$와 보상 함수 $r(s,a,s')$를 사전에 알아야 하는데, 실제 환경에서는 이를 정확히 알 수 없는 경우가 많다.

> [!done]- 정답
> 위 ①②가 모두 맞다. 이 두 한계를 극복하기 위해 **Model-free 강화학습** (Monte Carlo, TD Learning 등)이 등장하였다.

---

**Q5.** 다음 중 결정론적 상태 전이(Deterministic state-transition)에서 최적 행동 $\mu_*(s)$를 구하는 올바른 수식은?

① $\mu_*(s) = \sum_a q_*(s,a)$
② $\mu_*(s) = \arg\max_a \pi(a|s)$
③ $\mu_*(s) = \arg\max_a \{r(s,a,s') + \gamma v_*(s')\}$
④ $\mu_*(s) = \min_a V(s)$

> [!done]- 정답
> **③** $s' = f(s,a)$ (결정론적 전이)일 때, 최적 행동은 즉각적 보상과 다음 상태의 가치합을 최대화하는 행동이다.

# Quiz Q4 풀이 — 4-Grid World Bellman Optimality Equation

> **문제**: 2-Grid World를 확장한 4-Grid World 문제에 대하여, Bellman Optimality Equation을 유도하라.

---

## 1. 환경 설정 (Q3와 동일)

```
왼쪽 벽                                     오른쪽 벽
  -1   | L1 | L2 | L3 | L4 |   -1
 (벽)   [🤖]             [🍎]  (벽)
                        +1
                    (L3→L4 전이 시)
```

| 상태 | 행동 | 다음 상태 | 보상 $r$ |
|------|------|-----------|----------|
| L1 | LEFT | L1 (벽) | **-1** |
| L1 | RIGHT | L2 | 0 |
| L2 | LEFT | L1 | 0 |
| L2 | RIGHT | L3 | 0 |
| L3 | LEFT | L2 | 0 |
| L3 | RIGHT | L4 (사과) | **+1** |
| L4 | LEFT | L3 | 0 |
| L4 | RIGHT | L4 (벽) | **-1** |

- 정책: **최적 정책** $\pi_*$ (미지수)
- 할인율: $\gamma = 0.9$

---

## 2. Q3 vs Q4 핵심 차이

| 구분 | Q3 (Bellman Equation) | Q4 (Bellman Optimality Equation) |
|------|----------------------|----------------------------------|
| 정책 | 고정된 stochastic policy $\pi$ (LEFT/RIGHT = 0.5/0.5) | 최적 policy $\pi_*$ (미지수) |
| 수식 | $v_\pi(s) = \sum_a \pi(a\|s)[r + \gamma v_\pi(s')]$ | $v_*(s) = \max_a \sum_{s'} p(s'\|s,a)[r + \gamma v_*(s')]$ |
| 문제 유형 | 연립방정식 (선형) | **최적화 문제 (비선형 — max 연산 포함)** |
| 풀이 방법 | 선형 시스템 직접 풀이 | 최적 행동 가정 후 선형화 → Dynamic Programming |

---

## 3. Bellman Optimality Equation 도출

$$\boxed{v_*(s) = \max_a \sum_{s'} p(s'|s,a) \left\{ r(s,a,s') + \gamma v_*(s') \right\}}$$

각 상태에서 두 행동(LEFT / RIGHT)의 가치를 비교하여 **최대값**을 선택:

---

### 상태 L1

$$v_*(L1) = \max \begin{cases} \underbrace{-1 + 0.9\,v_*(L1)}_{a = \text{LEFT (벽 충돌)}} \\ \underbrace{0 + 0.9\,v_*(L2)}_{a = \text{RIGHT}} \end{cases}$$

---

### 상태 L2

$$v_*(L2) = \max \begin{cases} \underbrace{0 + 0.9\,v_*(L1)}_{a = \text{LEFT}} \\ \underbrace{0 + 0.9\,v_*(L3)}_{a = \text{RIGHT}} \end{cases}$$

---

### 상태 L3

$$v_*(L3) = \max \begin{cases} \underbrace{0 + 0.9\,v_*(L2)}_{a = \text{LEFT}} \\ \underbrace{1 + 0.9\,v_*(L4)}_{a = \text{RIGHT (사과 획득)}} \end{cases}$$

---

### 상태 L4

$$v_*(L4) = \max \begin{cases} \underbrace{0 + 0.9\,v_*(L3)}_{a = \text{LEFT}} \\ \underbrace{-1 + 0.9\,v_*(L4)}_{a = \text{RIGHT (벽 충돌)}} \end{cases}$$

---

## 4. 풀이 전략 — Optimal Action 가정

> **비선형 문제 풀이 방법**: 직관적으로 최적 행동을 가정(정성적 추론)한 뒤, 선형 연립방정식을 풀고, 가정이 옳은지 사후 검증.

### 직관적 추론

| 상태 | 가정한 최적 행동 | 이유 |
|------|----------------|------|
| L1 | **RIGHT** | 왼쪽은 벽(-1), 오른쪽(L2)이 사과(L4)에 더 가까움 |
| L2 | **RIGHT** | L3 > L1 (사과에 더 가까운 방향) |
| L3 | **RIGHT** | 바로 오른쪽이 사과 → +1 즉시 획득 |
| L4 | **LEFT** | 오른쪽은 벽(-1), 왼쪽(L3)에서 +1 반복 획득 가능 |

---

## 5. 선형 연립방정식 풀이

가정한 optimal action 대입:

$$\begin{cases} v_*(L1) = 0.9\,v_*(L2) & \cdots (1) \\ v_*(L2) = 0.9\,v_*(L3) & \cdots (2) \\ v_*(L3) = 1 + 0.9\,v_*(L4) & \cdots (3) \\ v_*(L4) = 0.9\,v_*(L3) & \cdots (4) \end{cases}$$

### Step 1: (3), (4) 연립

(4)를 (3)에 대입:

$$v_*(L3) = 1 + 0.9 \times 0.9\,v_*(L3) = 1 + 0.81\,v_*(L3)$$

$$v_*(L3)(1 - 0.81) = 1 \implies v_*(L3) = \frac{1}{0.19} = \frac{100}{19}$$

### Step 2: 역순 대입

$$v_*(L4) = 0.9 \times \frac{100}{19} = \frac{90}{19}$$

$$v_*(L2) = 0.9 \times \frac{100}{19} = \frac{90}{19}$$

$$v_*(L1) = 0.9 \times \frac{90}{19} = \frac{81}{19}$$

---

## 6. 최종 결과

$$\begin{cases} v_*(L1) = \dfrac{81}{19} \approx 4.2632 \\[10pt] v_*(L2) = \dfrac{90}{19} \approx 4.7368 \\[10pt] v_*(L3) = \dfrac{100}{19} \approx 5.2632 \\[10pt] v_*(L4) = \dfrac{90}{19} \approx 4.7368 \end{cases}$$

```
 상태:   L1       L2       L3       L4
 v*:   4.2632   4.7368   5.2632   4.7368
         ↑                 ↑
      (가장 낮음)        (가장 높음 — 사과 바로 왼쪽)
```

> **Q3 결과와 비교**: Q3에서는 모든 값이 음수였지만, optimal policy를 따르면 모두 양수! 정책 최적화의 효과가 극명하게 나타남.

| | Q3 ($v_\pi$, random policy) | Q4 ($v_*$, optimal policy) |
|---|---|---|
| $v(L1)$ | -1.8141 | **+4.2632** |
| $v(L2)$ | -1.1061 | **+4.7368** |
| $v(L3)$ | -0.6439 | **+5.2632** |
| $v(L4)$ | -1.4359 | **+4.7368** |

---

## 7. Optimal Action Value 계산 (2-Grid 예시 방법 적용)

슬라이드 p.31의 방법 그대로 적용 — $q_*(s,a) = \sum_{s'} p(s'|s,a)\{r + \gamma v_*(s')\}$:

### 상태 L1에서의 Action Value

$$q_*(L1, \text{LEFT}) = -1 + 0.9 \times v_*(L1) = -1 + 0.9 \times 4.2632 = -1 + 3.8368 = \mathbf{2.8368}$$

$$q_*(L1, \text{RIGHT}) = 0 + 0.9 \times v_*(L2) = 0.9 \times 4.7368 = \mathbf{4.2632}$$

$$\Rightarrow \mu_*(L1) = \text{RIGHT} \quad (4.2632 > 2.8368)$$

### 상태 L2에서의 Action Value

$$q_*(L2, \text{LEFT}) = 0 + 0.9 \times v_*(L1) = 0.9 \times 4.2632 = \mathbf{3.8368}$$

$$q_*(L2, \text{RIGHT}) = 0 + 0.9 \times v_*(L3) = 0.9 \times 5.2632 = \mathbf{4.7368}$$

$$\Rightarrow \mu_*(L2) = \text{RIGHT} \quad (4.7368 > 3.8368)$$

### 상태 L3에서의 Action Value

$$q_*(L3, \text{LEFT}) = 0 + 0.9 \times v_*(L2) = 0.9 \times 4.7368 = \mathbf{4.2632}$$

$$q_*(L3, \text{RIGHT}) = 1 + 0.9 \times v_*(L4) = 1 + 0.9 \times 4.7368 = 1 + 4.2632 = \mathbf{5.2632}$$

$$\Rightarrow \mu_*(L3) = \text{RIGHT} \quad (5.2632 > 4.2632)$$

### 상태 L4에서의 Action Value

$$q_*(L4, \text{LEFT}) = 0 + 0.9 \times v_*(L3) = 0.9 \times 5.2632 = \mathbf{4.7368}$$

$$q_*(L4, \text{RIGHT}) = -1 + 0.9 \times v_*(L4) = -1 + 0.9 \times 4.7368 = -1 + 4.2632 = \mathbf{3.2632}$$

$$\Rightarrow \mu_*(L4) = \text{LEFT} \quad (4.7368 > 3.2632)$$

---

## 8. Optimal Policy 정리

$$\pi_*(a|s) = \begin{cases} 1 & \text{if } a = \mu_*(s) \\ 0 & \text{otherwise} \end{cases}$$

| 상태 | $q_*(\cdot, \text{LEFT})$ | $q_*(\cdot, \text{RIGHT})$ | $\mu_*(s)$ |
|------|--------------------------|---------------------------|------------|
| L1 | 2.8368 | **4.2632** | **RIGHT** →  |
| L2 | 3.8368 | **4.7368** | **RIGHT** →  |
| L3 | 4.2632 | **5.2632** | **RIGHT** →  |
| L4 | **4.7368** | 3.2632 | **LEFT**  ←  |

```
이동 방향:  L1 →→→ L2 →→→ L3 →→→ L4 ←←← (L4에서 다시 왼쪽)
                                    🍎(+1)
```

> Optimal policy: L1, L2, L3에서는 오른쪽으로 이동하여 사과에 접근, L4에 도달하면 +1을 받고 다시 L3(왼쪽)로 이동. **L3↔L4 사이를 반복하며 +1 보상을 계속 수집**하는 것이 최적 전략.

---

## 9. 검증 (Verification)

| 방정식                              | 계산         | 기대값    | 일치  |
| -------------------------------- | ---------- | ------ | --- |
| $v_*(L1) = \max(2.8368, 4.2632)$ | **4.2632** | 4.2632 | ✅   |
| $v_*(L2) = \max(3.8368, 4.7368)$ | **4.7368** | 4.7368 | ✅   |
| $v_*(L3) = \max(4.2632, 5.2632)$ | **5.2632** | 5.2632 | ✅   |
| $v_*(L4) = \max(4.7368, 3.2632)$ | **4.7368** | 4.7368 | ✅   |

---

*참고: Bellman Optimality Equation은 비선형(max 연산) → 일반 연립방정식으로 직접 풀 수 없음. 실제 대규모 문제에서는 **Dynamic Programming** (Policy Iteration, Value Iteration) 기법으로 해결.*



# Q4 풀이 — 4-Grid World Bellman Optimality Equation 상세 해설

> **문제 (p.32)**: 앞 Example(2-Grid World)을 확장한 4-Grid World 문제에 대하여, Bellman Optimality Equation을 유도하라. ($\gamma = 0.9$)

---

# Part 1. Bellman Optimality Equation 이론 상세 설명

## 1-1. 핵심 개념 계층 구조

```
MDP (상태, 행동, 전이확률, 보상, 할인율)
   │
   ├── Policy π(a|s)  →  Value Function vπ(s), qπ(s,a)
   │                          │
   │                          └── Bellman Equation (연립방정식)
   │
   └── Optimal Policy π*  →  Optimal Value Function v*(s), q*(s,a)
                                       │
                                       └── Bellman Optimality Equation (최적화 문제)
```

---

## 1-2. Value Function vs Optimal Value Function

### (1) 일반 State-Value Function $v_\pi(s)$

$$v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} ;\middle|; S_t = s\right]$$

- 정책 $\pi$ **고정** 후 상태 $s$에서의 **기대 누적 수익**
- 정책에 따라 값이 달라짐 → $v_{\mu_1}(s) \neq v_{\mu_2}(s) \neq \cdots$

### (2) Optimal State-Value Function $v_*(s)$

$$v_*(s) = \max_\pi, v_\pi(s)$$

- **모든 가능한 정책 중 최대**인 state-value
- 어떤 정책도 $v_*(s)$보다 높은 가치를 줄 수 없음

### (3) Optimal Action-Value Function $q_*(s,a)$

$$q_*(s,a) = \max_\pi, q_\pi(s,a)$$

- **모든 정책 중 최대**인 action-value
- "상태 $s$에서 행동 $a$를 취했을 때 얻을 수 있는 **최대** 기대 수익"

---

## 1-3. Bellman Equation vs Bellman Optimality Equation 비교

|구분|Bellman Equation|Bellman Optimality Equation|
|---|---|---|
|**대상**|$v_\pi(s)$ (특정 정책의 가치)|$v_*(s)$ (최적 가치)|
|**수식**|$v_\pi(s) = \sum_{a,s'}\pi(a|s),p(s'|
|**핵심 연산**|$\sum_a \pi(a|s)$ (가중 평균)|
|**문제 유형**|**선형 연립방정식**|**비선형 최적화 문제**|
|**풀이 방법**|선형 시스템 직접 풀이|Dynamic Programming 필요|
|**정책**|고정된 정책 $\pi$ 사용|정책 미지수 (최적화로 결정)|

---

## 1-4. Bellman Optimality Equation 유도

### State-Value 버전

일반 Bellman Equation에서 출발:

$$v_\pi(s) = \sum_{a,s'} \pi(a|s), p(s'|s,a){r(s,a,s') + \gamma, v_\pi(s')}$$

**최적화 문제**: $v_\pi(s)$를 최대화하는 $\pi(a|s)$는 무엇인가?

각 행동 $a$에 대한 기대 수익 $Q(a) = \sum_{s'} p(s'|s,a){r + \gamma, v_*(s')}$ 로 정의하면:

- $\pi(a|s)$는 확률 분포 → $Q(a)$의 가중 평균은 최대 원소를 초과 불가
- 따라서 최적 정책은 $Q(a)$가 최대인 행동에만 확률 1 배정 (deterministic)

$$\boxed{v__(s) = \max_a \sum_{s'} p(s'|s,a){r(s,a,s') + \gamma, v__(s')}}$$

이때 최적 정책:

$$\pi__(a|s) = \begin{cases} 1 & \text{if } a = \mu__(s) \ 0 & \text{otherwise} \end{cases}, \quad \mu__(s) = \arg\max_a \sum_{s'}p(s'|s,a){r + \gamma v__(s')}$$

### Action-Value 버전

$$\boxed{q__(s,a) = \sum_{s'} p(s'|s,a)\left{r(s,a,s') + \gamma \max_{a'} q__(s',a')\right}}$$

### State-Value ↔ Action-Value 관계

$$v__(s) = \max_a, q__(s,a) \qquad q__(s,a) = \sum_{s'}p(s'|s,a){r + \gamma v__(s')}$$

---

## 1-5. 왜 비선형인가? — 풀이의 어려움

**일반 Bellman Eq. (선형)**:

$$v_\pi(s_1) = \pi_{\text{LEFT}} \cdot (-1 + 0.9 v_\pi(s_1)) + \pi_{\text{RIGHT}} \cdot (1 + 0.9 v_\pi(s_2))$$

정책 $\pi$ 고정 시 $v_\pi(s_1), v_\pi(s_2)$에 대한 **1차 연립방정식** → 직접 풀기 가능

**Bellman Optimality Eq. (비선형)**:

$$v__(s_1) = \max{-1 + 0.9 v__(s_1),\ 1 + 0.9 v_*(s_2)}$$

**max 연산** 포함 → 비선형 → 직접 연립방정식으로 풀 수 없음

**해결 전략**: 최적 행동 직관적 가정 → 선형화 → 풀이 → 사후 검증 (또는 Dynamic Programming)

---

# Part 2. p.30 참조 — 2-Grid World 예시 (Example 1)

## 2-1. 환경 설정

```
왼쪽 벽 | L1 | L2 | 오른쪽 벽
  -1    [🤖] [🍎]    -1
              +1
```

|전이|보상|
|---|---|
|L1 + LEFT → L1 (벽)|-1|
|L1 + RIGHT → L2 (사과)|+1|
|L2 + LEFT → L1|0|
|L2 + RIGHT → L2 (벽)|-1|

## 2-2. Bellman Optimality Equation

$$v__(L1) = \max \begin{cases} -1 + 0.9,v__(L1) \ +1 + 0.9,v__(L2) \end{cases}, \quad v__(L2) = \max \begin{cases} 0 + 0.9,v__(L1) \ -1 + 0.9,v__(L2) \end{cases}$$

## 2-3. 풀이: mu*(L1)=RIGHT, mu*(L2)=LEFT 가정 후 선형화

$$v__(L1) = 1 + 0.81,v__(L1) ;\Rightarrow; v__(L1) = \frac{100}{19} \approx 5.26, \quad v__(L2) = \frac{90}{19} \approx 4.74$$

## 2-4. Action Value 계산 (Optimal Policy 확인)

|상태|$q_*(\text{LEFT})$|$q_*(\text{RIGHT})$|$\mu_*$|
|---|:-:|:-:|:-:|
|L1|$-1+0.9\times5.26=3.73$|$+1+0.9\times4.74=\mathbf{5.26}$|RIGHT|
|L2|$0+0.9\times5.26=\mathbf{4.74}$|$-1+0.9\times4.74=3.26$|LEFT|

---

# Part 3. Q4 풀이 — 4-Grid World Bellman Optimality Equation

## 3-1. 환경 설정

```
왼쪽 벽                          오른쪽 벽
  -1   | L1 | L2 | L3 | L4 |   -1
 (벽)   [🤖]             [🍎]  (벽)
                          +1
                      (L3→L4 전이 시)
```

|상태|행동|다음 상태|보상|
|---|---|---|---|
|L1|LEFT|L1 (벽)|**-1**|
|L1|RIGHT|L2|0|
|L2|LEFT|L1|0|
|L2|RIGHT|L3|0|
|L3|LEFT|L2|0|
|L3|RIGHT|L4 (사과)|**+1**|
|L4|LEFT|L3|0|
|L4|RIGHT|L4 (벽)|**-1**|

---

## 3-2. Bellman Optimality Equation 도출 (핵심)

$$\boxed{v__(s) = \max_a \sum_{s'} p(s'|s,a){r(s,a,s') + \gamma, v__(s')}}$$

### 상태 L1

$$v__(L1) = \max \begin{cases} -1 + 0.9,v__(L1) & a = \text{LEFT} \ \phantom{-}0 + 0.9,v_*(L2) & a = \text{RIGHT} \end{cases} \quad \cdots (1)$$

### 상태 L2

$$v__(L2) = \max \begin{cases} 0 + 0.9,v__(L1) & a = \text{LEFT} \ 0 + 0.9,v_*(L3) & a = \text{RIGHT} \end{cases} \quad \cdots (2)$$

### 상태 L3

$$v__(L3) = \max \begin{cases} 0 + 0.9,v__(L2) & a = \text{LEFT} \ 1 + 0.9,v_*(L4) & a = \text{RIGHT} \end{cases} \quad \cdots (3)$$

### 상태 L4

$$v__(L4) = \max \begin{cases} \phantom{-}0 + 0.9,v__(L3) & a = \text{LEFT} \ -1 + 0.9,v_*(L4) & a = \text{RIGHT} \end{cases} \quad \cdots (4)$$

> 이 4개의 방정식이 **Bellman Optimality Equation** — max 연산 포함으로 비선형

---

## 3-3. Optimal State Value 풀이

### Step 1 — Optimal Action 가정

|상태|가정|근거|
|---|---|---|
|L1|RIGHT|왼쪽 벽 충돌(-1) 회피, 사과 방향|
|L2|RIGHT|L3가 L1보다 사과에 가까움|
|L3|RIGHT|바로 오른쪽 = 사과(+1 즉시 획득)|
|L4|LEFT|오른쪽 벽 충돌(-1) 회피, L3에서 +1 반복|

### Step 2 — 가정 하에 선형화

$$\begin{cases} v__(L1) = 0.9,v__(L2) \ v__(L2) = 0.9,v__(L3) \ v__(L3) = 1 + 0.9,v__(L4) \ v__(L4) = 0.9,v__(L3) \end{cases}$$

### Step 3 — 풀이

(4)를 (3)에 대입:

$$v__(L3) = 1 + 0.9 \times 0.9,v__(L3) = 1 + 0.81,v_*(L3)$$

$$\therefore; v_*(L3) = \frac{1}{1-0.81} = \frac{1}{0.19} = \frac{100}{19}$$

역대입:

$$v__(L4) = \frac{90}{19}, \quad v__(L2) = \frac{90}{19}, \quad v_*(L1) = \frac{81}{19}$$

### 결과

$$\boxed{ \begin{array}{|c|c|c|c|} \hline v__(L1) & v__(L2) & v__(L3) & v__(L4) \ \hline 81/19 \approx 4.26 & 90/19 \approx 4.74 & 100/19 \approx 5.26 & 90/19 \approx 4.74 \ \hline \end{array} }$$

---

## 3-4. Optimal Policy 도출 — Action Value 계산

$$q__(s,a) = \sum_{s'} p(s'|s,a){r(s,a,s') + \gamma, v__(s')}$$

### 상태 L1

$$\begin{cases} -1 + 0.9 \times 4.2632 = \mathbf{2.8368} & a = \text{LEFT} \ \phantom{-}0 + 0.9 \times 4.7368 = \mathbf{4.2632} & a = \text{RIGHT} \leftarrow \max \end{cases} \Rightarrow \mu_*(L1) = \text{RIGHT}$$

### 상태 L2

$$\begin{cases} 0 + 0.9 \times 4.2632 = \mathbf{3.8368} & a = \text{LEFT} \ 0 + 0.9 \times 5.2632 = \mathbf{4.7368} & a = \text{RIGHT} \leftarrow \max \end{cases} \Rightarrow \mu_*(L2) = \text{RIGHT}$$

### 상태 L3

$$\begin{cases} 0 + 0.9 \times 4.7368 = \mathbf{4.2632} & a = \text{LEFT} \ 1 + 0.9 \times 4.7368 = \mathbf{5.2632} & a = \text{RIGHT} \leftarrow \max \end{cases} \Rightarrow \mu_*(L3) = \text{RIGHT}$$

### 상태 L4

$$\begin{cases} 0 + 0.9 \times 5.2632 = \mathbf{4.7368} & a = \text{LEFT} \leftarrow \max \ -1 + 0.9 \times 4.7368 = \mathbf{3.2632} & a = \text{RIGHT} \end{cases} \Rightarrow \mu_*(L4) = \text{LEFT}$$

---

## 3-5. 최종 결과

|상태|$q_*(\text{LEFT})$|$q_*(\text{RIGHT})$|$\mu_*(s)$|$v_*(s)$|
|:-:|:-:|:-:|:-:|:-:|
|L1|2.8368|**4.2632**|→ RIGHT|**4.2632**|
|L2|3.8368|**4.7368**|→ RIGHT|**4.7368**|
|L3|4.2632|**5.2632**|→ RIGHT|**5.2632**|
|L4|**4.7368**|3.2632|← LEFT|**4.7368**|

```
이동:  L1 ──→ L2 ──→ L3 ──→ L4
       4.26   4.74   5.26   4.74
                      🍎(+1) │
               ←─────────────┘
         (L4에서 LEFT → L3으로 복귀 → 반복)
```

---

## 3-6. 검증

|상태|$\max(q_\text{LEFT},\ q_\text{RIGHT})$|$v_*(s)$|일치|
|:-:|:-:|:-:|:-:|
|L1|$\max(2.8368,\ 4.2632) = 4.2632$|4.2632|✅|
|L2|$\max(3.8368,\ 4.7368) = 4.7368$|4.7368|✅|
|L3|$\max(4.2632,\ 5.2632) = 5.2632$|5.2632|✅|
|L4|$\max(4.7368,\ 3.2632) = 4.7368$|4.7368|✅|

---

# Part 4. 핵심 공식 정리

|이름|공식|
|---|---|
|**Bellman Equation**|$v_\pi(s) = \sum_{a,s'}\pi(a\|s),p(s'\|s,a)[r + \gamma v_\pi(s')]$|
|**Bellman Optimality (State)**|$v__(s) = \max_a\sum_{s'}p(s'\|s,a)[r + \gamma v__(s')]$|
|**Bellman Optimality (Action)**|$q__(s,a) = \sum_{s'}p(s'\|s,a)[r + \gamma\max_{a'}q__(s',a')]$|
|**Optimal Action**|$\mu__(s) = \arg\max_a, q__(s,a)$|
|**관계식**|$v__(s) = \max_a, q__(s,a)$|

> Bellman Optimality Equation이 비선형(max 포함)이므로 대규모 문제는 **Dynamic Programming** (Value Iteration, Policy Iteration)으로 해결

---

_출처: 03-Markov Decision Process (2).pdf + Part2 p.30, p.32 참조_
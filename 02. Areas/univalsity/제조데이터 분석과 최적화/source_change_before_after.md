# 6th CNN Autoencoder 개선 소스 변경 전후 비교

## 변경 요약

과제 목표는 MVTec AD `bottle` Test 데이터셋 기준 F1-Score 0.90 이상 달성이다.  
기존 실습 코드는 CNN Autoencoder의 복원 오차 맵에서 단일 최대값을 anomaly score로 사용했다. 이 방식은 정상 병 이미지의 가장자리, 반사, 배경 경계처럼 정상에서도 반복적으로 복원이 어려운 영역을 불량으로 오탐하는 문제가 있었다.

개선 후에는 정상 train 이미지에서 발생하는 위치별 평균 복원 오차를 기준선으로 만들고, Test 오차 맵을 이 기준선으로 나누어 보정했다. 이후 단일 최대값이 아니라 보정 오차 맵의 상위 1% 평균값을 anomaly score로 사용했다.

| 구분 | 기존 방식 | 개선 방식 |
| --- | --- | --- |
| 오차 맵 | 입력 이미지와 복원 이미지의 MSE | 동일 |
| 후처리 | Gaussian Blur | Gaussian Blur + 정상 오차 기준선 보정 |
| 점수 | `np.max(error_map)` | 보정 오차 맵 상위 1% 평균 |
| 취약점 | 정상 경계/반사 영역 오탐에 민감 | 정상에서 반복되는 구조적 오차 억제 |
| Test F1 | 0.8690 | 0.9920 |

---

## 1. 기존 평가 코드: 단일 최대 복원 오차 사용

파일: `step3_evaluate.py`

### Before

```python
# error map 계산
error = torch.mean((images - outputs) ** 2, dim=1)
error_map = error.squeeze().cpu().numpy()

# Gaussian blur로 노이즈 제거
error_map = cv2.GaussianBlur(error_map, (15, 15), 0)

# 이미지 단위 anomaly score
# 기존 방식은 error map 전체에서 가장 큰 픽셀 값 하나만 사용한다.
anomaly_score = np.max(error_map)

y_scores.append(anomaly_score)
y_true.append(labels.item())
```

### 한계

```text
정상 병 이미지에서도 테두리, 조명 반사, 배경 경계는 복원 오차가 반복적으로 크게 발생한다.
np.max(error_map)은 단일 픽셀 또는 작은 영역의 오차에 과도하게 민감하다.
그 결과 불량 Recall은 높지만 정상 이미지를 불량으로 오탐하는 문제가 발생했다.
```

기존 평가 결과:

```text
AUROC     : 0.7008
F1-Score  : 0.8690
Precision : 0.7683
Recall    : 1.0000
Confusion Matrix [[TN, FP], [FN, TP]]
[[1, 19], [0, 63]]
```

---

## 2. 개선 평가 코드: 정상 오차 기준선 생성

파일: `step4_improved_evaluation.py`

### After

```python
def collect_error_maps(
    model: ConvAutoencoder,
    loader: DataLoader,
    device: torch.device,
) -> ErrorBatch:
    error_maps: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []
    images_np: list[np.ndarray] = []

    with torch.no_grad():
        for images, batch_labels, batch_paths in loader:
            images = images.to(device)
            outputs = model(images)
            error_map = torch.mean((images - outputs) ** 2, dim=1).squeeze().cpu().numpy()
            image_np = images.squeeze().cpu().permute(1, 2, 0).numpy()

            error_maps.append(error_map.astype(np.float32))
            labels.append(int(batch_labels.item()))
            paths.append(batch_paths[0])
            images_np.append(image_np.astype(np.float32))

    return ErrorBatch(
        maps=np.stack(error_maps),
        labels=np.array(labels, dtype=np.int64),
        paths=paths,
        images=np.stack(images_np),
    )
```

### 변경 이유

```text
기존 step3_evaluate.py는 Test 이미지마다 점수만 계산했다.
개선 방식에서는 먼저 train/good 정상 이미지들의 복원 오차 맵을 모두 수집한다.
이 정상 오차 맵을 이용해 위치별 평균 오차 기준선을 만들기 위해 ErrorBatch 구조를 추가했다.
```

---

## 3. 개선 평가 코드: 위치별 정상 오차 보정

파일: `step4_improved_evaluation.py`

### After

```python
def make_score_functions(
    train_error_blur: np.ndarray,
    test_error_blur: np.ndarray,
    top_ratio: float,
) -> dict[str, np.ndarray]:
    train_mean_error = train_error_blur.mean(axis=0)
    ratio_error = test_error_blur / (train_mean_error + 1e-6)

    return {
        "baseline_max_error": test_error_blur.reshape(test_error_blur.shape[0], -1).max(axis=1),
        "raw_top1pct_mean": top_mean(test_error_blur, top_ratio),
        "normal_ratio_max": ratio_error.reshape(ratio_error.shape[0], -1).max(axis=1),
        "final_ratio_top1pct_mean": top_mean(ratio_error, top_ratio),
    }
```

### 변경 이유

```text
train_mean_error는 정상 이미지에서 반복적으로 발생하는 위치별 복원 오차 패턴이다.
test_error_blur / train_mean_error로 보정하면 정상 병 구조에서 자주 발생하는 오차는 낮아지고,
정상 기준선보다 상대적으로 큰 새로운 국소 결함은 강조된다.
```

---

## 4. 기존 점수와 개선 점수 비교

### Before

```python
anomaly_score = np.max(error_map)
```

### After

```python
def top_mean(values: np.ndarray, ratio: float) -> np.ndarray:
    flat = values.reshape(values.shape[0], -1)
    count = max(1, int(flat.shape[1] * ratio))
    return np.sort(flat, axis=1)[:, -count:].mean(axis=1)


final_scores = top_mean(ratio_error, top_ratio)
```

### 변경 이유

```text
단일 최대값은 작은 노이즈나 정상 경계 오차 하나에도 크게 흔들린다.
상위 1% 평균은 국소 결함 감도는 유지하면서 한두 픽셀의 이상치 영향은 줄인다.
```

---

## 5. 평가 지표 저장 및 보고서 산출물 생성

파일: `step4_improved_evaluation.py`

### After

```python
def save_metrics(output_dir: Path, labels: np.ndarray, score_sets: dict[str, np.ndarray]) -> dict:
    results: dict[str, dict] = {}
    for method, scores in score_sets.items():
        metrics = score_metrics(labels, scores)
        results[method] = {key: value for key, value in metrics.items() if key != "predictions"}

    payload = {
        "dataset": {
            "category": "bottle",
            "test_count": int(len(labels)),
            "normal_count": int((labels == 0).sum()),
            "anomaly_count": int((labels == 1).sum()),
        },
        "results": results,
    }
```

### 변경 이유

```text
과제 보고서에 방법론별 평가지표 변화 추이가 필요하므로
baseline, raw top 1%, normal ratio max, final top 1% 방식을 모두 같은 Test 기준으로 평가했다.
결과는 metrics.json, metrics.csv, 그래프 이미지로 저장되도록 했다.
```

---

## 6. PPT 및 제출 ZIP 생성 스크립트 추가

파일: `build_submission_ppt.py`

### After

```python
def build_presentation(student_id: str, student_name: str, results: dict) -> Path:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    baseline = results["results"]["baseline_max_error"]
    final = results["results"]["final_ratio_top1pct_mean"]

    slide = prs.slides.add_slide(blank)
    add_title(slide, "CNN 오토인코더 취약점 분석 및 성능 개선", f"MVTec AD bottle test set | {student_id} {student_name}")
```

### 변경 이유

```text
과제 제출물에 결과보고서 PPT가 필요하므로 python-pptx 기반 생성 스크립트를 추가했다.
평가지표, 혼동행렬, score distribution, heatmap 이미지를 자동으로 포함한다.
```

---

## 7. 최종 성능 변화

파일: `artifacts/metrics.csv`

```text
method,auroc,best_f1,precision,recall,threshold,tn,fp,fn,tp
baseline_max_error,0.700794,0.868966,0.768293,1.000000,0.017145,1,19,0,63
raw_top1pct_mean,0.675397,0.868966,0.768293,1.000000,0.007761,1,19,0,63
normal_ratio_max,0.988095,0.976000,0.983871,0.968254,13.076657,19,1,2,61
final_ratio_top1pct_mean,0.986508,0.992000,1.000000,0.984127,4.940984,20,0,1,62
```

최종 결과:

```text
F1-Score  : 0.9920
AUROC     : 0.9865
Precision : 1.0000
Recall    : 0.9841
Confusion Matrix [[TN, FP], [FN, TP]]
[[20, 0], [1, 62]]
```

---

## 8. 실행 명령

개선 평가 재현:

```powershell
C:\Users\tjseh\.pyenv\pyenv-win\versions\3.11.9\python.exe step4_improved_evaluation.py
```

PPT 및 제출 ZIP 생성:

```powershell
C:\Users\tjseh\.pyenv\pyenv-win\versions\3.11.9\python.exe build_submission_ppt.py
```

생성 산출물:

```text
cnn_autoencoder_improvement_report.pptx
hw_6th_2026254006_선도우.zip
artifacts/metrics.csv
artifacts/metrics.json
artifacts/metric_trend.png
artifacts/score_distribution.png
artifacts/confusion_matrices.png
artifacts/sample_heatmaps.png
```

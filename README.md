# Two-Stage Feature Selection + SSAE+Attention 기반 NIDS

Stacked Sparse Autoencoder(SSAE) + Attention을 이용한 네트워크 침입 탐지(NIDS) 연구.
CIC-IDS2017, CSE-CIC-IDS2018 두 데이터셋으로 실험했으며, 한국콘텐츠학회논문지 투고를
목표로 현재 작성 중이다. 4학년 1학기부터 진행.

> 이 연구는 [SSAE 기반 특징 중요도 분석(SHAP)](../nids-shap-conference) 컨퍼런스
> 연구의 확장판이다. 1차 연구에서 "feature selection이 유의미하다"는 걸 확인한 뒤,
> feature selection 방법을 2단계로 정교화하고 모델에 Attention을 추가해 두 데이터셋으로
> 확장했다.

## 실험 환경

RTX 4080 SUPER / Ryzen 9 9900X3D / RAM 64GB / CUDA 11.8 / Python 3.10.19 / PyTorch 2.7.1

## 연구 질문

이 연구는 세 가지 질문에 답하는 구조로 설계했다.

- **RQ1.** SSAE+Attention이 다른 모델 구조보다 효과적인가?
- **RQ2.** 78개 feature를 22개로 줄여도 탐지 성능을 유지할 수 있는가?
- **RQ3.** 제안한 Consensus-22가 다른 22개 feature 선택 방법보다 효과적인가?

## Two-Stage Feature Selection

**Stage 1** — Full-78 → XGBoost와 ElasticNet 각각 적용 → 교집합 → Mutual Information 적용

| 데이터셋 | Stage 1 결과 |
|---|---|
| CIC-IDS2017 | 44개 |
| CSE-CIC-IDS2018 | 43개 |

**Stage 2** — Stage 1 결과 → XGBoost Permutation Importance와 ElasticNet Permutation
Importance 각각 적용 → 교집합 → **Consensus-22** (두 데이터셋 모두 최종 22개 feature)

## RQ1 — 모델 구조 비교 (Consensus-22 기준)

| Model | CIC-IDS2017 Acc | F1 | ROC-AUC | FAR | CSE-CIC-IDS2018 Acc | F1 | ROC-AUC | FAR |
|---|---|---|---|---|---|---|---|---|
| SAE+Attention | 0.9715 | 0.9310 | 0.9966 | 0.0291 | 0.9052 | 0.7985 | 0.8642 | 0.0001 |
| SSAE | 0.9724 | 0.9291 | 0.9971 | 0.0147 | 0.9080 | 0.8056 | 0.8286 | 0.0002 |
| **SSAE+Attention (제안)** | **0.9816** | **0.9541** | **0.9978** | 0.0160 | **0.9080** | **0.8056** | **0.8756** | 0.0001 |
| Transformer | 0.9606 | 0.9051 | 0.9940 | 0.0373 | 0.9078 | 0.8055 | 0.8512 | 0.0010 |

Full-78(전체 feature)에서도 동일하게 비교했을 때, 두 데이터셋 모두 SSAE+Attention이
근소하게 최고 성능을 기록했다 (2017: Acc 0.9847/F1 0.9616, 2018: Acc 0.9082/F1 0.8060).
→ **RQ1 결론: SSAE+Attention이 feature 수와 무관하게 가장 일관되게 우수하다.**

## RQ2 — Full-78 vs Consensus-22 (SSAE+Attention 고정)

| 지표 | 2017 Full-78 | 2017 Consensus-22 | 2018 Full-78 | 2018 Consensus-22 |
|---|---|---|---|---|
| Accuracy | 0.9847 | 0.9816 | 0.9082 | 0.9080 |
| F1 | 0.9616 | 0.9541 | 0.8060 | 0.8056 |
| ROC-AUC | 0.9986 | 0.9978 | 0.9175 | 0.8756 |
| Training Time | 83.80s | 80.36s | 505.17s | 467.40s |
| Memory | 1844.29MB | **1169.10MB** | 5853.23MB | **2392.95MB** |
| Parameters | 44,064 | **29,672** | 44,064 | **29,672** |

→ **RQ2 결론: feature를 78개에서 22개로(약 72%) 줄여도 성능 저하는 1%p 이내로 미미한
반면, 메모리 사용량은 2017에서 37%, 2018에서 59% 감소했다.** 성능 손실 대비 자원 절감
효과가 뚜렷하다.

## RQ3 — Feature 선택 방법 비교 (22개로 동일 조건, SSAE+Attention 고정)

"아무 22개나 뽑아도 비슷한 성능이 나오는 게 아니라, Consensus-22가 실제로 더 나은가?"를
검증하기 위한 실험.

**CIC-IDS2017**

| 방법 | Accuracy | F1 | ROC-AUC | FAR |
|---|---|---|---|---|
| Random-22 (5회 평균) | 0.9486 | 0.8505 | 0.9728 | 0.0217 |
| XGBoost-22 | 0.9772 | 0.9443 | 0.9951 | 0.0236 |
| ElasticNet-22 | 0.9809 | 0.9521 | 0.9980 | 0.0151 |
| MI-22 | 0.9727 | 0.9338 | 0.9964 | 0.0279 |
| **Consensus-22 (제안)** | **0.9816** | **0.9541** | **0.9978** | 0.0160 |

**CSE-CIC-IDS2018**

| 방법 | Accuracy | F1 | ROC-AUC | FAR |
|---|---|---|---|---|
| Random-22 (5회 평균) | 0.8338 | 0.6295 | 0.8091 | 0.0603 |
| XGBoost-22 | 0.9084 | 0.8065 | 0.9828 | 0.0001 |
| ElasticNet-22 | 0.9083 | 0.8062 | 0.9293 | 0.0001 |
| MI-22 | 0.8933 | 0.7673 | 0.9595 | 0.0002 |
| **Consensus-22 (제안)** | **0.9080** | **0.8056** | 0.8756 | 0.0001 |

→ **RQ3 결론: Random-22는 다른 방법들보다 확연히 낮고 변동성도 크다** (2018 Random 5회
결과가 0.7645~0.8831로 크게 흔들림 — Random Run 2는 특히 낮게 나와, 무작위 선택이 얼마나
불안정한지 보여준다). **Consensus-22는 단일 알고리즘 방법들과 대등하거나 근소 우위**를
보였다 — 성능을 크게 해치지 않으면서 두 알고리즘의 합의를 반영한 feature set이라는 점에서
의의가 있다.

## 남은 작업

- 혼동행렬(Confusion Matrix) 분석 — Consensus-22 기준 2017/2018 각각의 TP/TN/FP/FN 분석 예정
- CSE-CIC-IDS2018 Random-22 Run 2가 유독 낮게(Acc 0.7645) 나온 원인 확인 필요
- 논문 작성 진행 중 (미투고)

## 폴더 구조

- `preprocessing/` — 데이터 전처리
- `feature_selection/` — Two-Stage Feature Selection (Stage1/Stage2, RQ3용 baseline 포함)
- `models/` — SSAE, SSAE+Attention, SAE+Attention, Transformer
- `evaluation/` — 혼동행렬, SHAP 분석 등
- `feature_lists/` — 각 방법으로 선택된 feature 목록 텍스트
- `checkpoints/` — 학습된 모델 가중치 (조합별)

## 실행 방법

```bash
# 1. 전처리
python preprocessing/preprocess.py        # 2017
python preprocessing/preprocess_2018.py   # 2018

# 2. Feature Selection (Stage 1 → Stage 2)
python feature_selection/XGBoost_top22.py
python feature_selection/ElasticNet_top22.py
python feature_selection/intersection.py      # Stage 1: 교집합 + MI
python feature_selection/XGBoost_PI.py
python feature_selection/ElasticNet_PI.py
python feature_selection/pi_intersection.py   # Stage 2: PI 교집합 → Consensus-22

# 3. 모델 학습/평가
python "models/SSAE+Attention.py"
```

> 원본 CIC-IDS CSV와 전처리된 `.pt` 텐서(수십 GB)는 용량 문제로 레포에 포함하지 않았다.

## 배운 점

- 두 개의 서로 다른 feature importance 알고리즘(XGBoost/ElasticNet)의 교집합을 2단계에
  걸쳐 적용해, 단일 알고리즘이나 무작위 선택보다 안정적인 feature set(Consensus-22)을
  구성했다. Random 선택이 5회 반복에서 큰 변동성(Acc 0.76~0.97)을 보인 것과 대비하여,
  이 방법론의 필요성을 실험으로 입증했다.
- 정확도뿐 아니라 메모리 사용량, 파라미터 수, 학습/추론 시간까지 측정해, feature 축소가
  실제로 자원 절감에 기여하는지를 정량적으로 확인했다.
- 연구 질문(RQ)을 먼저 설계하고 그에 맞는 실험을 체계적으로 쌓아가는 논문 작성 프로세스를
  경험했다.

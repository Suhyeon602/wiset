# Shapley 기반 AdaLoRA Rank 배분 for Viewport Prediction

**NetLLM** viewport prediction 코드베이스 위에서, **AdaLoRA**의 rank 배분을
**Shapley value 기반 동적 배분**으로 대체한 연구입니다. Llama-2-7B 백본을 360°
비디오 viewport 예측에 파인튜닝하되, 세 가지 parameter-efficient 방법을 **동일
파라미터 예산**으로 비교합니다.

| 방법 | Rank 배분 방식 | 위치 |
|------|---------------|------|
| **LoRA** | 모든 target 모듈에 고정 rank | `run_plm.py` + `models/low_rank.py` |
| **AdaLoRA (sensitivity)** | 동적 — sensitivity score로 중요도 산정 | `run_adalora.py` (기본) |
| **AdaLoRA (Shapley)** | 동적 — 온라인 Shapley value로 중요도 산정 | `run_adalora.py --use-shapley` |

핵심 연구 질문: **각 rank의 기여도를 Shapley value로 추정하면, AdaLoRA 기본
sensitivity 휴리스틱보다 LoRA 예산을 더 잘 배분할 수 있는가?**

---

## 방법론

AdaLoRA는 각 가중치 업데이트를 SVD 형태 `ΔW = P Λ Q`로 파라미터화하고, 덜
중요하다고 판단한 특이값(`lora_E`)을 pruning하여 그 예산을 다른 곳에 재배분합니다.
기본 AdaLoRA는 중요도를 **sensitivity 휴리스틱**(magnitude × gradient, smoothing)으로
매깁니다.

이 프로젝트는 그 scoring을 **온라인 Shapley value**로 교체합니다. 각 rank/모듈을
협력 게임의 player로 보고, coalition `S`의 가치 `v(S)`를 (음의) validation loss로
정의합니다. 순열(permutation)을 샘플링해 각 성분의 Shapley 기여도를 추정하고, rank
allocator가 sensitivity 대신 **Shapley 중요도**로 pruning/유지를 결정합니다.

- Shapley 추정은 **동적/온라인**입니다 — one-shot이 아니라 AdaLoRA 재배분
  스케줄(`deltaT`)에 맞춰 반복 계산됩니다.
- `v(S) = -평균 validation loss` (소수의 held-out 배치로 계산).
- 추정 정확도와 비용을 조절하는 두 노브:
  `--shapley-perm`(샘플링할 순열 수), `--shapley-batches`(각 coalition 평가에 쓸
  배치 수).

주요 파일:

- `run_adalora.py` — AdaLoRA 메인 드라이버(train + test). PLM을 PEFT AdaLoRA
  모델로 감싸고, `--use-shapley`가 켜지면 Shapley loss function을 rank
  allocator에 연결합니다.
- `shapley_rank.py` — `v(S)`(coalition → validation loss) 구성 및 loss-fn 주입.
- `shapley_allocator.py` — `ShapleyRankAllocator(RankAllocator)`,
  `mask_to_budget`를 오버라이드하여 Shapley 중요도로 배분.
- `models/low_rank_adalora.py` — AdaLoRA target 모듈 맵 + `AdaLoraConfig`.
- `models/low_rank.py` — plain-LoRA target 모듈 맵 + `LoraConfig`.
- `run_plm.py` — plain-LoRA 드라이버(train + test), LoRA baseline용.
- `adalora_rank_heatmap.py` — 최종 (모듈, 레이어)별 rank 추출 및 히트맵 시각화.
- `evaluate_result.py` — 결과 CSV 사후 비교.

---

## Target 모듈

LoRA와 AdaLoRA 모두 **동일한** attention projection을 Llama-2-7B의 **32개 레이어
전부**에 붙이므로, 비교가 parameter-matched 됩니다.

```python
# models/low_rank.py  와  models/low_rank_adalora.py
TARGET_MODULES['llama'] = ["q_proj", "v_proj"]
```

`q_proj`와 `v_proj`만 adapt합니다(고전적 최소 LoRA 구성). `k_proj`, `o_proj`,
그리고 MLP(`gate_proj`, `up_proj`, `down_proj`)는 frozen 상태로 둡니다. rank 4
기준 약 2.1M 학습 파라미터(7B 백본의 약 0.09%)입니다.

> 더 많은 모듈을 adapt하려면 리스트를 확장하세요(예: `k_proj`, `o_proj`,
> `gate_proj`, `up_proj`, `down_proj` 추가). **두 파일 모두** 바꿔야 LoRA와
> AdaLoRA가 matched 상태를 유지하며, 모든 방법을 재학습해야 합니다.

---

## 환경

Llama-2-7B의 rank-4 파인튜닝은 VRAM ≥ 24 GB 단일 GPU면 충분합니다(학습 중 peak
약 26 GB, inference/test 약 25 GB). RTX 4090(48 GB)에서 개발했습니다.

```bash
conda create -n netllm python=3.10 -y
conda activate netllm

# PyTorch (드라이버에 맞는 CUDA 빌드)
pip install torch==2.2.0

# 고정 의존성 (버전 민감 — 임의로 올리지 말 것)
pip install transformers==4.40.2
pip install "huggingface_hub<1.0"          # 0.23.4 검증됨; 반드시 <1.0
pip install peft==0.6.2
pip install "numpy<2"
pip install pandas scipy matplotlib
```

> 버전 주의: `huggingface_hub`는 **반드시** `<1.0`이어야 모델 로딩이 깨지지
> 않습니다. `peft==0.6.2`는 이 코드가 대상으로 하는 AdaLoRA API 버전입니다.
> `numpy<2`는 고정된 torch와의 ABI 충돌을 피하기 위함입니다.

---

## 모델

백본: **Llama-2-7B**. 가중치는 로컬 디렉토리(`../downloaded_plms/llama/base`)에서
로드합니다. 게이트가 없는
[`NousResearch/Llama-2-7b-hf`](https://huggingface.co/NousResearch/Llama-2-7b-hf)
미러는 access token 없이 받을 수 있습니다.

```bash
huggingface-cli download NousResearch/Llama-2-7b-hf \
  --local-dir downloaded_plms/llama/base \
  --local-dir-use-symlinks False
```

경로는 `config.py`의 `cfg.plms_dir`로 해석됩니다
(`plms_dir = ../downloaded_plms`). 레이아웃:

```
downloaded_plms/
└── llama/
    └── base/          # Llama-2-7B safetensors + config + tokenizer
```

---

## 데이터셋

360° 비디오 viewport trajectory 데이터셋. `dataset/load_dataset.py`의
`create_dataset(...)`로 로드합니다. 본 실험 설정:

| 항목 | 값 |
|------|-----|
| 데이터셋 | `Jin2022` |
| 주파수 | 5 Hz (`--dataset-frequency 5`) |
| History window | 10 (`--his-window 10`) |
| Future(예측) window | 20 (`--fut-window 20`) |
| Sample step | 15 (`--sample-step 15`) |
| Test split 크기 | 1698 샘플, 126개 (video, user) 그룹, 6개 비디오 |

데이터셋은 `config.py`에 설정된 경로에 위치합니다. 원본 viewport 로그를 trim하고
windowing하는 과정은 원본 NetLLM 데이터 준비 절차를 참고하세요.

---

## 실행

모든 명령은 작업 디렉토리를 `viewport_prediction/`으로, 단일 GPU(`cuda:0`)를
가정합니다. 메모리 단편화를 줄이기 위해 allocator 환경변수를 설정하세요.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
```

아래 모든 run의 공통 하이퍼파라미터:
`--epochs 4 --bs 1 --grad-accum-steps 32 --lr 5e-4 --rank 4 --seed 1
--scheduled-sampling`.

### 1. LoRA baseline (train + test)

```bash
python run_plm.py --adapt --test \
  --train-dataset Jin2022 --test-dataset Jin2022 \
  --dataset-frequency 5 --sample-step 15 --his-window 10 --fut-window 20 \
  --plm-type llama --plm-size base \
  --epochs 4 --bs 1 --grad-accum-steps 32 --lr 5e-4 \
  --rank 4 --seed 1 --scheduled-sampling --device cuda:0
```

### 2. AdaLoRA — sensitivity (기본)

```bash
python run_adalora.py --adapt --test \
  --train-dataset Jin2022 --test-dataset Jin2022 \
  --dataset-frequency 5 --sample-step 15 --his-window 10 --fut-window 20 \
  --plm-type llama --plm-size base \
  --epochs 4 --bs 1 --grad-accum-steps 32 --lr 5e-4 \
  --rank 4 --seed 1 --scheduled-sampling --device cuda:0
```

### 3. AdaLoRA — Shapley

```bash
python run_adalora.py --adapt --test \
  --train-dataset Jin2022 --test-dataset Jin2022 \
  --dataset-frequency 5 --sample-step 15 --his-window 10 --fut-window 20 \
  --plm-type llama --plm-size base \
  --epochs 4 --bs 1 --grad-accum-steps 32 --lr 5e-4 \
  --rank 4 --seed 1 --scheduled-sampling --device cuda:0 \
  --use-shapley --shapley-perm 3 --shapley-batches 4
```

### Test만 실행 (학습된 checkpoint 재사용)

`--model-path`로 특정 `best_model` 디렉토리를 지정하고 `--adapt`를 뺍니다.
Shapley checkpoint면 결과 파일이 올바르게 태깅되도록 `--use-shapley`를 유지하세요.

```bash
python run_adalora.py --test \
  --train-dataset Jin2022 --test-dataset Jin2022 \
  --dataset-frequency 5 --sample-step 15 --his-window 10 --fut-window 20 \
  --plm-type llama --plm-size base \
  --epochs 4 --bs 1 --grad-accum-steps 32 --lr 5e-4 \
  --rank 4 --seed 1 --scheduled-sampling --device cuda:0 \
  --use-shapley \
  --model-path "data/ft_plms/llama_base_adalora/freeze_plm_False/Jin2022/5Hz/his_10_fut_20_ss_15_epochs_4_bs_32_lr_0.0005_seed_1_rank_4_scheduled_sampling_True_shapley/best_model"
```

### AdaLoRA 전용 플래그

| 플래그 | 의미 | 기본값 |
|--------|------|--------|
| `--use-shapley` | sensitivity 대신 Shapley 중요도 사용 | off |
| `--shapley-perm` | Shapley 추정당 샘플링할 순열 수 | 3 |
| `--shapley-batches` | 각 coalition 평가에 쓸 val 배치 수 | 4 |
| `--grad-ckpt` | gradient checkpointing 활성화 (VRAM 절약) | off |
| `--skip-eval` | test 후 `evaluate_result.py` 단계 건너뛰기 | off |

> 단일 GPU 주의: 두 job을 동시에 돌리지 마세요. 학습 중 Shapley 재배분/검증
> 구간에서 peak가 약 47 GB까지 튀기 때문에, 그 타이밍에 두 번째 프로세스가 7B
> 가중치를 로드하면 OOM이 납니다. 순차적으로 실행하세요.

---

## 결과 저장 위치

파인튜닝된 어댑터 (`best_model/` = best validation checkpoint):

```
data/ft_plms/
├── llama_base_low_rank/freeze_plm_False/Jin2022/5Hz/<prefix>/best_model            # LoRA
└── llama_base_adalora/ freeze_plm_False/Jin2022/5Hz/<prefix>_original/best_model    # sensitivity AdaLoRA
                                                     /<prefix>_shapley/best_model     # Shapley AdaLoRA
```

Test 결과 CSV 및 rank 히트맵:

```
data/results/
├── llama_base_low_rank/freeze_plm_False/Jin2022/5Hz/<prefix>_results.csv
└── llama_base_adalora/ freeze_plm_False/Jin2022/5Hz/<prefix>_original_results.csv
                                                     /<prefix>_shapley_results.csv
                                                     /<prefix>_shapley_adalora_ranks.png
```

여기서
`<prefix> = his_10_fut_20_ss_15_epochs_4_bs_32_lr_0.0005_seed_1_rank_4_scheduled_sampling_True`.

`_original` / `_shapley` suffix는 run 모드에 따라 자동으로 붙으므로, sensitivity와
Shapley run이 서로를 덮어쓰지 않습니다. LoRA는 별도의 `_low_rank` 트리에
저장됩니다.

### 결과 CSV 형식

각 `*_results.csv`는 가로로 긴 형태입니다 — 행이 `video, user, mae, rmse`이고
열이 1698개 테스트 샘플입니다. 로드 방법:

```python
import pandas as pd
df = pd.read_csv(path, index_col=0).T     # -> 샘플당 한 행
# columns: video, user, mae, rmse
```

---

## 현재 결과 (rank 4, Jin2022 / 5 Hz, seed 1)

**LoRA vs Shapley-AdaLoRA** (n = 1698, 126개 (video, user) 그룹):

| 지표 | LoRA | Shapley-AdaLoRA |
|------|------|-----------------|
| MAE (mean) | 14.379 | **14.242** |
| MAE (median) | 10.403 | **10.188** |
| RMSE (mean) | 22.988 | **22.638** |
| RMSE (median) | 14.334 | **12.982** |

전체 차이는 통계적으로 유의하지 않지만(Mann-Whitney: MAE p = 0.87, RMSE
p = 0.22; 그룹평균 paired Wilcoxon: MAE p = 0.83, RMSE p = 0.27), 네 개 집계
지표 전부에서 Shapley가 우세합니다.

**비디오별 MAE** — 전체 평균이 숨긴 이질성이 드러납니다:

| video | n | LoRA | Shapley | Δ | p |
|-------|-----|-------|---------|-----|------|
| 4 | 294 | **12.712** | 12.977 | +0.266 | 0.180 |
| 8 | 294 | **11.758** | 12.514 | +0.756 | **0.008** |
| 14 | 294 | 10.982 | **10.645** | −0.337 | 0.182 |
| 18 | 228 | 14.907 | **14.591** | −0.316 | 0.650 |
| 24 | 294 | **12.123** | 12.529 | +0.406 | 0.362 |
| 25 | 294 | 23.909 | **22.272** | −1.637 | **0.021** |

Shapley는 **가장 어려운 비디오(25)에서 유의한 개선**을, 쉬운 비디오(8)에서 유의한
손해를 보입니다. 효과가 콘텐츠 난이도를 따라갑니다. (6개 비디오에 대한 Bonferroni
보정 시 α = 0.0083이므로 video 8만 유지됩니다 — 개별 비디오 p보다 난이도-이득
경향을 주 결과로 보고하세요.)

> 3-way 비교(LoRA vs **sensitivity**-AdaLoRA vs Shapley-AdaLoRA)가 핵심
> 헤드라인입니다. sensitivity-AdaLoRA 열은 해당 run이 완료되면 추가됩니다.

---

## 분석 재현

```bash
python evaluate_result.py     # 방법 간 결과 CSV 비교
```

`evaluate_result.py`는 LoRA와 AdaLoRA 결과 CSV를 읽어 비디오별/그룹별 차이를
보고합니다. 특정 방법의 CSV가 없으면 그 방법 열에서 `KeyError`가 납니다 — 모든
방법을 먼저 실행하거나, 존재하는 CSV만 가리키도록 스크립트를 조정하세요.

---

## Acknowledgements

**NetLLM** viewport prediction 프레임워크 위에 구축되었습니다. 백본 가중치는
Meta의 Llama-2(NousResearch HF 미러 경유). AdaLoRA와 LoRA는 Hugging Face
**PEFT**로 구현되었습니다.

## License

<!-- TODO: 라이선스 명시(예: MIT). 단, Llama-2 가중치는 Meta 커뮤니티
     라이선스를, NetLLM 코드는 자체 라이선스를 따르므로 재배포 전 둘 다
     확인하세요. -->

#!/usr/bin/env bash
# train_both_gpus.sh
# original(sensitivity AdaLoRA) 와 shapley(online Shapley AdaLoRA) 를
# 각각 다른 GPU 에서 동시에 [학습 → 테스트 → evaluate_result.py] 자동 실행한다.
#
# 핵심 설계:
#   - 두 프로세스는 CUDA_VISIBLE_DEVICES 로 물리 GPU 를 격리하고,
#     둘 다 --device cuda:0 을 쓴다(각자 자기 GPU 만 cuda:0 으로 본다).
#   - 공통 인자(COMMON_ARGS)는 반드시 동일해야 통제비교가 성립한다.
#     차이는 오직 --use-shapley 유무뿐이다.
#   - --adapt --test 를 함께 주므로 한 프로세스 안에서 학습 후 곧바로
#     테스트하고, 테스트 끝에 run_adalora.py 가 evaluate_result.py 를 자동 호출한다.
#   - shapley run 은 학습상태 + coalition forward 라 peak 메모리가 더 크다.
#     20GB 카드에서 shapley 가 아슬아슬하면 SHAPLEY_PERM/SHAPLEY_BATCHES 를 낮추고
#     --grad-ckpt 를 켠다(아래 SHAPLEY_EXTRA 에 이미 포함).
#
# 전제: 각 run 의 모델이 GPU 1장(20GB)에 들어간다.
#   llama2-7B FP16(~13.5GB 가중치) → 20GB 1장에 학습 가능(original ~16~18GB).
#   shapley 는 스파이크로 ~18~20GB 경계라 --grad-ckpt 로 헤드룸 확보 권장.

set -u

# ─────────────────────────────────────────────────────────────
# GPU 배정 (물리 GPU 번호)
# ─────────────────────────────────────────────────────────────
GPU_ORIGINAL=0
GPU_SHAPLEY=1

# ─────────────────────────────────────────────────────────────
# Shapley 비용/메모리 knob — 20GB 에서 OOM 나면 더 낮춰라
# ─────────────────────────────────────────────────────────────
SHAPLEY_PERM=3        # Monte-Carlo 순열 수 (작을수록 빠름/가벼움)
SHAPLEY_BATCHES=4     # v(S) 계산에 쓸 고정 valid 배치 수 (작을수록 스파이크↓)

# shapley run 에만 추가로 붙일 인자 (헤드룸 확보용 gradient checkpointing 포함)
SHAPLEY_EXTRA="--use-shapley --shapley-perm ${SHAPLEY_PERM} --shapley-batches ${SHAPLEY_BATCHES} --grad-ckpt"

# ─────────────────────────────────────────────────────────────
# 공통 학습/테스트 인자 — ↓↓↓ 네가 쓰던 run_adalora.py 커맨드에 맞춰 검증할 것 ↓↓↓
#   (--device / --use-shapley 는 여기 넣지 말 것. 자동 처리됨)
#   --adapt --test 둘 다 있으므로 학습→테스트→평가가 한 번에 돈다.
# ─────────────────────────────────────────────────────────────
COMMON_ARGS=(
  --adapt
  --test
  --train-dataset  Jin2022
  --test-dataset   Jin2022
  --dataset-frequency 5
  --sample-step    15
  --his-window     10
  --fut-window     20
  --plm-type       llama
  --plm-size       base          # NetLLM 의 llama2-7B 경로명에 맞게 조정
  --epochs         4
  --bs             1
  --grad-accum-steps 32
  --lr             5e-4
  --rank           4
  --seed           1
  --scheduled-sampling
  --device         cuda:0        # 각 프로세스는 자기 GPU 만 cuda:0 으로 본다
  # --freeze-plm                 # 네 원래 커맨드에 있었으면 주석 해제
  # --multimodal                 # 네 원래 커맨드에 있었으면 주석 해제
  # --eval-script /path/to/evaluate_result.py   # 기본은 run_adalora.py 와 같은 폴더
)

# ─────────────────────────────────────────────────────────────
# 메모리 단편화로 인한 가짜 OOM 완화 (두 프로세스 공통)
# ─────────────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs
LOG_ORIG="logs/original_${TS}.log"
LOG_SHAP="logs/shapley_${TS}.log"

echo "[launch] original → GPU ${GPU_ORIGINAL}   shapley → GPU ${GPU_SHAPLEY}"
echo "[launch] rank=4  seed=1  (두 run 공통) / adapt+test+evaluate 자동 체인"

# ── original (sensitivity AdaLoRA) ─────────────────────────────
CUDA_VISIBLE_DEVICES=${GPU_ORIGINAL} \
  python run_adalora.py "${COMMON_ARGS[@]}" \
  > "${LOG_ORIG}" 2>&1 &
PID_ORIG=$!
echo "  original  PID=${PID_ORIG}  log=${LOG_ORIG}"

# ── shapley (online Shapley AdaLoRA) ───────────────────────────
CUDA_VISIBLE_DEVICES=${GPU_SHAPLEY} \
  python run_adalora.py "${COMMON_ARGS[@]}" ${SHAPLEY_EXTRA} \
  > "${LOG_SHAP}" 2>&1 &
PID_SHAP=$!
echo "  shapley   PID=${PID_SHAP}  log=${LOG_SHAP}"

echo "[info] 로그 실시간 확인:  tail -f ${LOG_ORIG}  또는  tail -f ${LOG_SHAP}"
echo "[info] 두 파이프라인(학습+테스트+평가)이 끝날 때까지 대기 중..."

# 둘 다 끝날 때까지 대기하고 각각의 종료코드 리포트
wait ${PID_ORIG}; RC_ORIG=$?
wait ${PID_SHAP}; RC_SHAP=$?

echo "[done] original rc=${RC_ORIG}   shapley rc=${RC_SHAP}"
if [ ${RC_ORIG} -ne 0 ] || [ ${RC_SHAP} -ne 0 ]; then
  echo "[warn] 종료코드 0 이 아닌 run 이 있다. 위 로그 확인 (OOM 이면 shapley knob 낮추기)."
  exit 1
fi
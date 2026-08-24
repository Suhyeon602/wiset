"""
adalora_interaction_probe.py
================================================================
AdaLoRA triplet 간 "상호작용(interaction)"이 실재하는지를 확인하는 진단 도구.
목적: sensitivity -> Shapley 교체가 의미를 가지려면 triplet들이 non-additive
      (서로 중복/보완)해야 한다. 이 스크립트는 그 전제를 데이터로 검증한다.

구성
  Tier 0  analyze_checkpoint(path)      : 학습된 P,Q,Lambda 행렬만으로 진단 (forward 0회)
  Tier 1  compute_loo(model, loss_fn)   : 각 triplet exact leave-one-out (forward n회)
  Tier 1  interaction_report(df)        : LOO vs magnitude vs sensitivity 상관/redundancy 비율
  Tier 2  pairwise_interaction(...)     : 의심 쌍에 대한 2차 차분 g_ij

peft AdaLoRA 파라미터 규약 (peft 0.6.2 기준)
  lora_A : (r, in_features)   = Q  (right singular vectors, 행이 삼중항)
  lora_B : (out_features, r)  = P  (left  singular vectors, 열이 삼중항)
  lora_E : (r, 1)             = diag(Lambda)  (singular values; pruned 삼중항은 0)

해석 주의
  * redundancy 의 직접 증거 = slack (||P^T P - I||_F) 와 pairwise cosine.
  * stable rank 는 직교가 유지되면 redundancy 가 아니라 Lambda 스펙트럼의
    집중도를 잰다. 직교가 깨질수록(=slack 큼) redundancy 성분이 섞여 sr 이 내려간다.
    따라서 sr 은 보조 지표로 읽고, 판정은 slack + cosine 으로 한다.
"""

import re
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 공통 유틸
# ----------------------------------------------------------------------
LORA_TAGS = ("lora_A", "lora_B", "lora_E")


def _to_numpy(x):
    """torch.Tensor / np.ndarray 모두 받아 float64 numpy 로."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _module_name(key: str) -> str:
    """'...c_attn.lora_A.default' 또는 '...c_attn.lora_A' -> '...c_attn'."""
    for tag in LORA_TAGS:
        if f".{tag}." in key:          # ...lora_A.default (접미사 있음)
            return key.split(f".{tag}.")[0]
        if key.endswith(f".{tag}"):    # ...lora_A (접미사 없음) ← 이 케이스가 빠져 있었음
            return key[: -len(f".{tag}") - 1]
    return key


def _short(name: str) -> str:
    """긴 module 경로를 'h.{layer}.{attn|mlp}.{proj}' 로 축약(히트맵 라벨과 맞춤)."""
    m = re.search(r"h\.(\d+)\.(attn|mlp)\.(\w+)", name)
    return f"h{m.group(1)}.{m.group(2)}.{m.group(3)}" if m else name


def stable_rank(dW: np.ndarray) -> float:
    s = np.linalg.svd(dW, compute_uv=False)
    return float((s ** 2).sum() / (s.max() ** 2 + 1e-12))


# ======================================================================
# Tier 0 : checkpoint 만으로 (forward 0회)  --- 여기부터 먼저 돌린다
# ======================================================================
def load_adapter_state_dict(path: str) -> dict:
    """
    adapter_model.bin / .safetensors / 전체 체크포인트 어디서든
    lora_A/B/E 텐서를 뽑아 {key: ndarray} 로 반환.
    torch 없이도 .safetensors 는 numpy 로 읽는다.
    """
    if path.endswith(".safetensors"):
        from safetensors.numpy import load_file
        raw = load_file(path)
    else:
        import torch
        raw = torch.load(path, map_location="cpu")
        # 전체 체크포인트라면 state_dict 를 찾아 들어간다
        if isinstance(raw, dict) and "state_dict" in raw:
            raw = raw["state_dict"]
    return {k: _to_numpy(v) for k, v in raw.items()
            if any(f".{t}." in k or k.endswith(f".{t}") for t in LORA_TAGS)}


def _group_modules(sd: dict) -> dict:
    mods = defaultdict(dict)
    for k, v in sd.items():
        for tag in LORA_TAGS:
            if f".{tag}." in k or k.endswith(f".{tag}"):
                mods[_module_name(k)][tag] = v
    return mods


def analyze_checkpoint(path: str, tol: float = 1e-8) -> pd.DataFrame:
    """
    module 별로 orthogonality slack, stable rank, pairwise cosine 을 계산.
    상호작용(중복)이 '실재할 수 있는지'의 1차 필터.
    """
    sd = load_adapter_state_dict(path)
    mods = _group_modules(sd)
    if not mods:
        raise ValueError("lora_A/B/E 텐서를 찾지 못함. 키 규약을 확인하라.")

    rows = []
    for base, d in mods.items():
        if not all(t in d for t in LORA_TAGS):
            continue
        Q = d["lora_A"]                 # (r, in)
        P = d["lora_B"]                 # (out, r)
        E = d["lora_E"].reshape(-1)     # (r,)

        surv = np.abs(E) > tol          # 살아남은 삼중항만
        r_full, r_s = E.shape[0], int(surv.sum())
        if r_s == 0:
            continue
        Ps, Qs, Es = P[:, surv], Q[surv, :], E[surv]

        # --- redundancy 직접 지표: orthogonality slack ---
        I = np.eye(r_s)
        slack_P = float(np.linalg.norm(Ps.T @ Ps - I))   # 열 직교(왼쪽 특이벡터)
        slack_Q = float(np.linalg.norm(Qs @ Qs.T - I))   # 행 직교(오른쪽 특이벡터)

        # --- pairwise cosine (surviving 삼중항 간) ---
        def max_mean_cos(M, axis_vectors):
            # axis_vectors: 각 열(또는 행)이 한 삼중항의 벡터
            V = axis_vectors / (np.linalg.norm(axis_vectors, axis=0, keepdims=True) + 1e-12)
            C = np.abs(V.T @ V)
            np.fill_diagonal(C, 0.0)
            return float(C.max()), float(C.sum() / (r_s * (r_s - 1) + 1e-12))
        maxcos_P, meancos_P = max_mean_cos(None, Ps)            # P 열 벡터
        maxcos_Q, meancos_Q = max_mean_cos(None, Qs.T)         # Q 행 -> 열로

        # --- 보조 지표: stable rank (Lambda 집중도 + redundancy 혼합) ---
        dW = Ps @ np.diag(Es) @ Qs
        sr = stable_rank(dW)

        rows.append(dict(
            module=_short(base),
            r_surv=r_s, r_init=r_full,
            slack_P=slack_P, slack_Q=slack_Q,
            maxcos=max(maxcos_P, maxcos_Q),
            meancos=max(meancos_P, meancos_Q),
            stable_rank=sr, sr_over_r=sr / r_s,
            lam_max=float(np.abs(Es).max()), lam_min=float(np.abs(Es).min()),
        ))

    df = pd.DataFrame(rows).sort_values("slack_P", ascending=False).reset_index(drop=True)
    _print_tier0(df)
    return df


def _print_tier0(df: pd.DataFrame):
    pd.set_option("display.width", 160, "display.max_columns", 20)
    print("\n" + "=" * 78)
    print("TIER 0  |  checkpoint-only interaction probe")
    print("=" * 78)
    show = df[["module", "r_surv", "slack_P", "slack_Q",
               "maxcos", "meancos", "stable_rank", "sr_over_r"]].copy()
    for c in ["slack_P", "slack_Q", "maxcos", "meancos", "stable_rank", "sr_over_r"]:
        show[c] = show[c].map(lambda x: f"{x:.3f}")
    print(show.to_string(index=False))

    # 요약 판정
    hi_slack = df[df["slack_P"] > 0.5]
    hi_cos = df[df["maxcos"] > 0.3]
    print("\n판정 힌트")
    print(f"  * slack_P > 0.5  인 module: {len(hi_slack)}/{len(df)}  "
          f"(직교 붕괴 = 삼중항 상관 채널 존재)")
    print(f"  * maxcos  > 0.3  인 module: {len(hi_cos)}/{len(df)}  "
          f"(중복 후보 쌍 존재)")
    if len(hi_slack) == 0 and len(hi_cos) == 0:
        print("  => 신호 약함. 이 regime 에서는 additive 에 가까움 -> 여기서 멈춰도 됨.")
    else:
        print("  => 신호 있음. Tier 1 (exact LOO) 로 진행 가치 있음.")
        print("     특히 다음 module 을 우선 조사:",
              ", ".join(df.sort_values("maxcos", ascending=False)["module"].head(3)))
    print("주의: stable_rank/sr_over_r 는 Lambda 집중도도 섞인 보조 지표. "
          "redundancy 판정은 slack/maxcos 로.")


# ======================================================================
# Tier 1 : exact leave-one-out  (forward n회)  --- Tier0 통과 시
# ======================================================================
def _iter_svd_layers(model):
    """peft AdaLoRA 의 SVDLinear 계층(=lora_E 를 가진 module)들을 순회."""
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_E"):
            yield name, mod


def compute_loo(model, loss_fn, adapter="default", tol=1e-8, verbose=True):
    """
    각 삼중항 i 를 mask(lambda_i -> 0) 하고 eval loss 변화를 측정.
        LOO_i = L_eval(N∖{i}) - L_eval(N)            (양수 = i 가 중요)

    인자
      model    : 로드된 peft AdaLoRA 모델 (eval() 상태 권장)
      loss_fn  : loss_fn(model) -> float.  *반드시 고정 eval 배치*에서 스칼라 반환.
                 (viewport prediction 이면 MAE 를 그대로 넣어도 된다)
    반환
      DataFrame[module, idx, lambda, loo]  (surviving 삼중항만)
    """
    import torch

    base = float(loss_fn(model))          # v(N)
    if verbose:
        print(f"[LOO] baseline L_eval(N) = {base:.6f}")

    records = []
    for name, mod in _iter_svd_layers(model):
        E = mod.lora_E[adapter]           # (r,1) Parameter
        with torch.no_grad():
            evec = E.detach().clone()
        r = evec.shape[0]
        for i in range(r):
            lam = float(evec[i, 0])
            if abs(lam) <= tol:           # 이미 pruned
                continue
            with torch.no_grad():
                E[i, 0] = 0.0             # mask
            loss_i = float(loss_fn(model))
            with torch.no_grad():
                E[i, 0] = lam             # restore
            records.append(dict(module=_short(name), idx=i,
                                **{"lambda": abs(lam)}, loo=loss_i - base))
        if verbose:
            print(f"[LOO] {_short(name):20s} done ({r} triplets)")

    return pd.DataFrame(records)


def attach_sensitivity(df_loo, sensitivity_dict):
    """
    선택: AdaLoRA RankAllocator 가 들고 있는 삼중항 sensitivity s(G_i)=Ibar*Ubar 를
    {(module_short, idx): score} 형태로 넘기면 열로 붙인다. (축 B 비교용)
    """
    df = df_loo.copy()
    df["sensitivity"] = df.apply(
        lambda r: sensitivity_dict.get((r["module"], int(r["idx"])), np.nan), axis=1)
    return df


def interaction_report(df, q=0.25):
    """
    Tier 1 핵심 판정.
      축 A (redundancy) : lambda 상위 q  &  loo 하위 q  인 삼중항 비율
      축 B (선형화 오차) : sensitivity vs loo Spearman  (sensitivity 있을 때)
      축 C (분포)        : loo 의 heavy-tail 여부(truncated MC 실행성)
    """
    from scipy.stats import spearmanr

    print("\n" + "=" * 78)
    print("TIER 1  |  exact leave-one-out interaction report")
    print("=" * 78)
    n = len(df)
    lam, loo = df["lambda"].to_numpy(), df["loo"].to_numpy()

    # --- 축 A: redundancy ratio ---
    hi_lam = lam >= np.quantile(lam, 1 - q)
    lo_loo = loo <= np.quantile(loo, q)
    redund = df[hi_lam & lo_loo]
    ratio = len(redund) / max(hi_lam.sum(), 1)
    print(f"[축 A] magnitude vs LOO")
    print(f"       Spearman(lambda, LOO) = {spearmanr(lam, loo).correlation:+.3f}")
    print(f"       'lambda 상위{int(q*100)}% 인데 LOO 하위{int(q*100)}%' = "
          f"{len(redund)}/{int(hi_lam.sum())}  (redundancy 비율 {ratio:.2f})")
    if ratio > 0.15:
        print("       => 모델이 투자했으나 빼도 안 아픈 삼중항 다수 = redundancy 실재. "
              "Shapley 가 이길 여지 있음.")
    else:
        print("       => redundancy 신호 약함.")

    # --- 축 B: 선형화 오차 ---
    if "sensitivity" in df and df["sensitivity"].notna().any():
        m = df["sensitivity"].notna()
        rho = spearmanr(df.loc[m, "sensitivity"], df.loc[m, "loo"]).correlation
        print(f"[축 B] sensitivity vs LOO  Spearman = {rho:+.3f}  "
              f"(낮을수록 AdaLoRA 1차 근사가 부정확 = exact-LOO 만으로도 개선 여지)")

    # --- 축 C: 분포 ---
    pos = np.clip(loo, 0, None)
    share_top10 = np.sort(pos)[::-1][:max(1, n // 10)].sum() / (pos.sum() + 1e-12)
    print(f"[축 C] LOO 상위10% 삼중항이 총 LOO 의 {share_top10:.0%} 차지  "
          f"(높을수록 heavy-tail = truncated MC 유리)")

    print("\n다음 단계: 축 A 가 유의미하면 Tier 2 (pairwise_interaction) 로 "
          "cosine 높은 쌍을 확증.")
    return dict(redund_ratio=ratio, top10_share=float(share_top10),
                redund_triplets=redund[["module", "idx"]].values.tolist())


# ======================================================================
# Tier 2 : pairwise 2차 차분 g_ij  (표적: 의심 쌍만)
# ======================================================================
def pairwise_interaction(model, loss_fn, pairs, adapter="default"):
    """
    후보 쌍 [(module_name, i, j), ...] 에 대해
        g_ij = [v(N)-v(N\\i)] - [v(N\\j)-v(N\\{i,j})]
    g<0 redundancy(대체재), g>0 synergy(보완재), g~0 무상호작용.
    pairs 의 module_name 은 named_modules 의 *풀네임* 이어야 한다.
    """
    import torch

    def L(masked):  # masked: list of (module, idx)
        saved = []
        with torch.no_grad():
            for name, idx in masked:
                E = dict(model.named_modules())[name].lora_E[adapter]
                saved.append((E, idx, float(E[idx, 0]))); E[idx, 0] = 0.0
        val = float(loss_fn(model))
        with torch.no_grad():
            for E, idx, v in saved:
                E[idx, 0] = v
        return val

    vN = float(loss_fn(model))
    out = []
    for name, i, j in pairs:
        v_i = L([(name, i)]); v_j = L([(name, j)]); v_ij = L([(name, i), (name, j)])
        g = (vN - v_i) - (v_j - v_ij)
        kind = "redundancy" if g < -1e-6 else "synergy" if g > 1e-6 else "additive"
        out.append(dict(module=_short(name), i=i, j=j, g=g, kind=kind))
        print(f"  {_short(name):20s} ({i:>2},{j:>2})  g={g:+.5f}  {kind}")
    return pd.DataFrame(out)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AdaLoRA triplet interaction probe (Tier 0)")
    ap.add_argument("checkpoint",
                    help="adapter_model.bin / .safetensors / 전체 체크포인트 경로 "
                         "(epoch4 = rank32 도달분을 쓸 것)")
    ap.add_argument("--csv", default=None, help="Tier 0 결과 저장 경로(옵션)")
    args = ap.parse_args()

    df = analyze_checkpoint(args.checkpoint)
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\n[saved] {args.csv}")

    print("\nTier 1 을 돌리려면 이 파일을 import 해서:")
    print("  from adalora_interaction_probe import compute_loo, interaction_report")
    print("  df_loo = compute_loo(model, loss_fn)   # loss_fn = 고정 eval 배치 MAE")
    print("  interaction_report(df_loo)")
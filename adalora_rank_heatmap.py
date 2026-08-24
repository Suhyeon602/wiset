#!/usr/bin/env python3
"""
AdaLoRA 최종 rank 히트맵.

구성:
  1) extract_final_ranks(model) : 학습된 PEFT AdaLoRA 모델에서
       (weight 타입, layer) 별 '최종 rank'를 추출한다.
       최종 rank = 해당 모듈 lora_E(특이값 벡터)에서 0이 아닌 성분 수
       (= RankAllocator가 pruning 후 남긴 차원 수).
  2) plot_rank_heatmap(df, ...) : 참조 그림과 같은 스타일의 히트맵 저장.

모델을 이 환경에서 못 불러오면, 서버(mcnl-238)의 eval 코드에서 model이
메모리에 올라와 있는 지점에 아래만 넣으면 됩니다:

    from adalora_rank_heatmap import extract_final_ranks, plot_rank_heatmap
    df = extract_final_ranks(model)     # 표로 확인
    print(df)
    plot_rank_heatmap(df, out="adalora_ranks.png")
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams


# ── 1. 최종 rank 추출 ────────────────────────────────────────────────────

# 모듈 이름(정규식) → 행 라벨. 위에서부터 먼저 매칭.
# GPT-2는 attn이 c_attn(q,k,v fused) 하나라 W_{qkv}로 잡히고,
# q_proj/k_proj/v_proj 처럼 분리된 구조면 W_q/W_k/W_v로 잡힌다.
DEFAULT_TYPE_PATTERNS = [
    (r"q_proj",       "W_q"),
    (r"k_proj",       "W_k"),
    (r"v_proj",       "W_v"),
    (r"o_proj",       "W_o"),
    (r"gate_proj",    "W_{f1}"),
    (r"up_proj",      "W_{f2}"),
    (r"down_proj",    "W_{f3}"),
    (r"attn\.c_attn", "W_{qkv}"),   # GPT-2: fused QKV
    (r"attn\.c_proj", "W_o"),
    (r"mlp\.c_fc",    "W_{f1}"),
    (r"mlp\.c_proj",  "W_{f2}"),
]

LAYER_RE = re.compile(r"\.(?:h|layers|layer|block|blocks)\.(\d+)\.")


def _match_type(name, patterns):
    for pat, label in patterns:
        if re.search(pat, name):
            return label
    return None


def extract_final_ranks(model, adapter="default",
                        patterns=DEFAULT_TYPE_PATTERNS, one_indexed=True):
    """PEFT AdaLoRA 모델 → (type × layer) 최종 rank DataFrame."""
    records = []
    for name, module in model.named_modules():
        E = getattr(module, "lora_E", None)
        if E is None:
            continue
        # lora_E는 보통 ParameterDict(adapter 이름 keyed)
        if hasattr(E, "keys"):
            if adapter not in E:
                continue
            E = E[adapter]
        e = E.detach().float().reshape(-1)
        final_rank = int((e.abs() > 0).sum().item())

        ltype = _match_type(name, patterns)
        m = LAYER_RE.search(name + ".")
        if ltype is None or m is None:
            continue
        layer = int(m.group(1)) + (1 if one_indexed else 0)
        records.append((ltype, layer, final_rank))

    if not records:
        raise RuntimeError(
            "lora_E를 가진 모듈을 못 찾음. AdaLoRA 모델이 맞는지, "
            "adapter 이름이 'default'인지, target_modules 정규식을 확인하세요."
        )

    df = (pd.DataFrame(records, columns=["type", "layer", "rank"])
            .pivot_table(index="type", columns="layer",
                         values="rank", aggfunc="first"))
    return df


# ── 2. 히트맵 ────────────────────────────────────────────────────────────

# 위 → 아래 (참조 그림: W_f2 맨 위, W_q 맨 아래)
DEFAULT_ROW_ORDER = ["W_{f2}", "W_{f1}", "W_o", "W_v", "W_k", "W_q",
                     "W_{qkv}", "W_{f3}"]


def plot_rank_heatmap(df, out="adalora_rank_heatmap.png", vmax=None,
                      row_order=None, cmap="YlGn",
                      cbar_label="The final rank", dpi=200):
    df = df.copy()
    df = df[sorted(df.columns)]                       # layer 오름차순
    order = [r for r in (row_order or DEFAULT_ROW_ORDER) if r in df.index]
    order += [r for r in df.index if r not in order]  # 미지정 라벨은 뒤에
    df = df.loc[order]

    data = df.values.astype(float)
    if vmax is None:
        vmax = np.nanmax(data)

    rcParams["font.family"] = "serif"
    rcParams["mathtext.fontset"] = "cm"

    n_row, n_col = data.shape
    fig, ax = plt.subplots(figsize=(0.62 * n_col + 2.4, 0.62 * n_row + 1.0))
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")

    # 흰색 셀 경계
    ax.set_xticks(np.arange(-0.5, n_col, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_row, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    # 셀 값 주석 (배경 밝기에 따라 글자색 자동)
    thresh = vmax * 0.55
    for i in range(n_row):
        for j in range(n_col):
            v = data[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{int(round(v))}", ha="center", va="center",
                    color="white" if v > thresh else "0.15", fontsize=10)

    ax.set_xticks(np.arange(n_col))
    ax.set_xticklabels([str(c) for c in df.columns])
    ax.set_yticks(np.arange(n_row))
    ax.set_yticklabels([f"${r}$" for r in df.index], fontsize=14)
    ax.set_xlabel("Layer", fontsize=14)
    ax.tick_params(which="major", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(cbar_label, fontsize=12)
    cbar.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


# ── 데모(스타일 미리보기) ────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rows = ["W_{f2}", "W_{f1}", "W_o", "W_v", "W_k", "W_q"]
    layers = list(range(1, 13))
    # 초반 layer는 rank 낮고 중후반은 높은, 흔한 AdaLoRA 패턴을 흉내
    base = np.linspace(3, 11, len(layers))
    demo = np.clip(np.round(base + rng.normal(0, 1.6, (len(rows), len(layers)))),
                   0, 12).astype(int)
    df = pd.DataFrame(demo, index=rows, columns=layers)
    plot_rank_heatmap(df, out="adalora_ranks_6.png",
                      row_order=["W_{f2}", "W_{f1}", "W_o", "W_v", "W_k", "W_q"])
    print(df)
    print("mean rank:", df.values.mean())
    print("saved demo_adalora_rank_heatmap.png")
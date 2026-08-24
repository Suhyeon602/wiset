"""
shapley_rank.py
================================================================
AdaLoRA 삼중항 importance 를 sensitivity(s=Ī·Ū) 대신 Shapley value 로 계산한다.

핵심 substrate 는 v(S) = "삼중항 부분집합 S 만 켰을 때의 성능".
이건 run_adalora.py 의 validate() 와 똑같은 forward 를 고정 배치에서 돌린 것뿐이다.

제공 함수
  make_loss_fn(...)        고정 eval 배치에서 -MAE(=v) 반환하는 zero-arg 콜백
  iter_svd_layers(model)   lora_E 가진 SVDLinear 계층 순회
  shapley_importance(...)  truncated MC permutation Shapley (offline)
  compare_importances(...) Shapley vs magnitude(|λ|) 비교 (Spearman + top-b 겹침)
  run_shapley_probe(...)   test() 안에서 부를 드라이버

주의 (읽고 시작할 것)
  * rank32 전역 Shapley 는 삼중항 3072개라 불가능. 이 스크립트는 기본적으로
    '지정한 module 안에서만' Shapley 를 재고 나머지 module 은 full 로 켜둔다
    (=이 module 의 어떤 삼중항을 남길지의 조건부 Shapley). 전역/저예산 실험은
    rank 2~4 로 학습한 체크포인트에서 modules=None 으로 돌릴 때만 현실적.
  * masking = lora_E[adapter][i,0]=0. peft SVDLinear forward 가 lora_E 를 그대로
    쓰므로 이 한 줄이 삼중항을 끄는 것과 동일하다.
"""

import math
import numpy as np
import torch


# ----------------------------------------------------------------------
# 1) v(S) substrate : 고정 배치 loss (validate() 와 동일 forward)
# ----------------------------------------------------------------------
def make_loss_fn(pipeline, dataloader, args, n_batches=4):
    """
    dataloader 에서 n_batches 개를 '한 번' 떠서 고정한다. 반환된 콜백은
    매 호출마다 그 고정 배치에서 pipeline forward(teacher_forcing=False)를
    돌려 평균 loss(float)를 준다. 낮을수록 좋음 → v(S) = -loss.

    validate() 를 그대로 옮긴 것이라 정규화·teacher_forcing 설정이 학습과 일치.
    """
    from utils.normalize import normalize_data

    fixed = []
    for i, (history, future, vinfo) in enumerate(dataloader):
        if i >= n_batches:
            break
        history = normalize_data(history.to(args.device), args.train_dataset)
        future  = normalize_data(future.to(args.device),  args.train_dataset)
        fixed.append((history, future, vinfo))
    if not fixed:
        raise RuntimeError("dataloader 에서 배치를 못 떴다.")
    print(f"[shapley] fixed eval batches = {len(fixed)} (bs={args.bs})")

    @torch.no_grad()
    def loss_fn():
        pipeline.eval()
        tot = 0.0
        for history, future, vinfo in fixed:
            loss = pipeline(history, future, vinfo, teacher_forcing=False)
            tot += float(loss.item())
        return tot / len(fixed)

    return loss_fn


# ----------------------------------------------------------------------
# 2) SVDLinear 계층 순회 + 삼중항 mask
# ----------------------------------------------------------------------
def iter_svd_layers(model, adapter="default"):
    """(module_name, lora_E Parameter[(r,1)]) 를 순회. lora_E 있는 계층만."""
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_E"):
            E = mod.lora_E[adapter] if hasattr(mod.lora_E, "__getitem__") else mod.lora_E
            yield name, E


def collect_triplets(model, adapter="default", tol=1e-8):
    """
    살아있는(λ≠0) 삼중항 목록과 원본 λ 스냅샷.
    반환: players=[(name,i)...], orig={(name,i):float}, Eref={name:Parameter}
    """
    players, orig, Eref = [], {}, {}
    for name, E in iter_svd_layers(model, adapter):
        Eref[name] = E
        col = E.detach().reshape(-1)
        for i in range(col.shape[0]):
            v = float(col[i])
            if abs(v) > tol:
                players.append((name, i)); orig[(name, i)] = v
    return players, orig, Eref


def make_value_fn(loss_fn, players, orig, Eref, scope_players):
    """
    value_fn(active) -> v = -loss.  active ⊆ scope_players 만 토글하고,
    scope 밖 삼중항은 항상 원본(full)로 켜둔다 (조건부 Shapley).
    scope_players 가 전체 players 면 전역 Shapley.
    """
    scope = set(scope_players)

    @torch.no_grad()
    def set_state(active):
        active = set(active)
        for (name, i) in scope:            # scope 안: active 면 원본, 아니면 0
            Eref[name][i, 0] = orig[(name, i)] if (name, i) in active else 0.0
        # scope 밖은 건드리지 않음(항상 원본 유지)

    def value_fn(active):
        set_state(active)
        return -loss_fn()

    def restore():                          # 실험 후 원상복구
        with torch.no_grad():
            for (name, i) in scope:
                Eref[name][i, 0] = orig[(name, i)]

    return value_fn, restore


# ----------------------------------------------------------------------
# 3) truncated MC permutation Shapley  (test_shapley.py 로 검증된 로직)
# ----------------------------------------------------------------------
def shapley_permutation(scope_players, value_fn, n_perm, truncate_tol,
                        rng, v_full, v_empty, verbose=True):
    phi = {p: 0.0 for p in scope_players}
    n = len(scope_players)
    for t in range(n_perm):
        perm = list(scope_players); rng.shuffle(perm)
        active, v_prev = set(), v_empty
        for p in perm:
            if abs(v_full - v_prev) < truncate_tol:
                marg = 0.0                        # truncation
            else:
                active.add(p)
                v_cur = value_fn(frozenset(active))
                marg = v_cur - v_prev; v_prev = v_cur
            phi[p] += marg
        if verbose:
            print(f"[shapley] perm {t+1}/{n_perm} done", flush=True)
    return {p: phi[p] / n_perm for p in scope_players}


def shapley_importance(model, loss_fn, modules=None, adapter="default",
                       n_perm=8, truncate_frac=0.01, seed=0, verbose=True):
    """
    modules=None  -> 전역 Shapley(저예산에서만 현실적)
    modules=[...] -> 그 module 들 삼중항에 대해서만(나머지 full). module 이름은
                     named_modules 풀네임의 접미(예: 'h23.attn.c_attn')로 매칭.
    truncate_frac : |v_full-v_empty| 의 이 비율보다 개선이 작아지면 truncate.
    반환: {(module_name, idx): phi}
    """
    players, orig, Eref = collect_triplets(model, adapter)
    if modules is None:
        scope = players
    else:
        scope = [(n, i) for (n, i) in players if any(m in n for m in modules)]
    if not scope:
        raise RuntimeError("scope 에 삼중항이 없다. module 이름 매칭 확인.")
    if verbose:
        print(f"[shapley] players in scope = {len(scope)} "
              f"(total surviving = {len(players)})")

    value_fn, restore = make_value_fn(loss_fn, players, orig, Eref, scope)
    try:
        v_full  = value_fn(frozenset(scope))     # scope 전부 켬
        v_empty = value_fn(frozenset())          # scope 전부 끔
        tol = abs(v_full - v_empty) * truncate_frac
        if verbose:
            print(f"[shapley] v_full={v_full:.5f} v_empty={v_empty:.5f} "
                  f"trunc_tol={tol:.5f}")
        rng = np.random.default_rng(seed)
        phi = shapley_permutation(scope, value_fn, n_perm, tol, rng,
                                  v_full, v_empty, verbose)
    finally:
        restore()                                # 반드시 원상복구
    return phi, orig


# ----------------------------------------------------------------------
# 4) Shapley vs magnitude(|λ|) 비교 — "다른 삼중항을 고르나?"
# ----------------------------------------------------------------------
def compare_importances(phi, orig, keep_ratio=0.5):
    """
    module 별로 Shapley 와 |λ| 가 남길 top-k 삼중항이 얼마나 겹치는지.
    겹침이 낮을수록 Shapley 가 magnitude 와 '다른 결정'을 한다는 직접 증거.
    """
    from scipy.stats import spearmanr
    import pandas as pd
    from collections import defaultdict

    by_mod = defaultdict(list)
    for (name, i), s in phi.items():
        by_mod[name].append((i, s, abs(orig[(name, i)])))

    rows = []
    for name, lst in by_mod.items():
        idx = [t[0] for t in lst]
        sh  = np.array([t[1] for t in lst])
        mag = np.array([t[2] for t in lst])
        r = len(idx); k = max(1, int(round(r * keep_ratio)))
        keep_sh  = set(np.array(idx)[np.argsort(-sh)[:k]])
        keep_mag = set(np.array(idx)[np.argsort(-mag)[:k]])
        jac = len(keep_sh & keep_mag) / len(keep_sh | keep_mag)
        rho = spearmanr(sh, mag).correlation if r > 2 else float("nan")
        rows.append(dict(module=name.split("transformer.")[-1], r=r, keep_k=k,
                         spearman_sh_mag=round(rho, 3),
                         topk_overlap=round(jac, 3),
                         swaps=k - len(keep_sh & keep_mag)))
    df = pd.DataFrame(rows).sort_values("topk_overlap")
    print("\n=== Shapley vs magnitude(|λ|) : top-k 유지집합 비교 ===")
    print(df.to_string(index=False))
    print("\ntopk_overlap 낮음 = Shapley 가 magnitude 와 다른 삼중항 유지 "
          "= 지표 교체가 실제 할당을 바꿈. 1.0 이면 둘이 동일 결정(교체 무의미).")
    return df


# ----------------------------------------------------------------------
# 5) test() 안에서 부르는 드라이버
# ----------------------------------------------------------------------
def run_shapley_probe(pipeline, dataloader, args,
                      target_modules=None, n_batches=4, n_perm=8,
                      keep_ratio=0.5):
    """
    test() 에서 load_model 직후 호출.
    target_modules 예: ['h23.attn.c_attn','h0.mlp.c_fc','h22.attn.c_attn']
                       None 이면 전역(저예산 체크포인트에서만).
    """
    loss_fn = make_loss_fn(pipeline, dataloader, args, n_batches=n_batches)
    base = loss_fn()
    print(f"[shapley] baseline loss (all triplets on) = {base:.6f}")
    phi, orig = shapley_importance(pipeline.plm, loss_fn,
                                   modules=target_modules, n_perm=n_perm)
    df = compare_importances(phi, orig, keep_ratio=keep_ratio)
    return phi, df
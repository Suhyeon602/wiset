"""
shapley_allocator.py  (2-stage hybrid, speed-optimized)
================================================================
peft 0.6.2 의 RankAllocator 를 상속해, 삼중항 importance 를 sensitivity(s=Ī·Ū)
대신 Shapley value 로 계산하는 ShapleyRankAllocator.

────────────────────────────────────────────────────────────────
2-STAGE HYBRID (group_by="hybrid", 기본값)
  stage-1 : 모듈 단위 Shapley (빠름)
            player = 모듈. 모듈을 통째로 켜고/끄며 phi_module 계산.
            → players 512→수십, antithetic+truncation 으로 추가 가속.
  stage-2 : 모듈 예산 배분 + 모듈 내부 |λ| top-k
            전체 budget 을 phi_module(양수만) 에 비례 배분해 각 모듈의
            정수 rank(=몇 개 삼중항 남길지) 를 정하고, 그 모듈 안에서는
            |λ| 상위 k 개만 남긴다.
            → rank 가 8/5/3/0 처럼 연속적으로 나온다 (module grouping 의
              "8 아니면 0" 이진 배분 문제 해결). 원조 AdaLoRA heatmap 형태.

  이 방식은 mask_to_budget 을 완전히 오버라이드한다(peft 전역 threshold 대신
  모듈별 예산→내부 top-k 로 직접 마스킹).

기타 group_by 옵션 (비교/디버그용):
  "module"  : stage-1 만. 모듈 통째 on/off (rank = 8 or 0).
  "triplet" : 전삼중항 개별 Shapley (느림, baseline).
────────────────────────────────────────────────────────────────

v(S) = -loss_fn():  '고정 validation 배치' forward 평균 loss 콜백. run() 에서 주입.
비용: mask_ind=True 인 스텝(실제 재분배)에서만 계산.
"""

import numpy as np
import torch
from peft.tuners.adalora.layer import RankAllocator


class ShapleyRankAllocator(RankAllocator):
    def __init__(self, model, peft_config, adapter_name,
                 loss_fn=None, n_perm=3, truncate_frac=0.05, seed=0,
                 group_by="hybrid", antithetic=True, verbose=True):
        """
        group_by : "hybrid"  → 2단계 (모듈 Shapley + 내부 |λ| top-k). 연속 rank.
                   "module"  → 모듈 통째 on/off (rank 8 or 0).
                   "triplet" → 전삼중항 Shapley (느림, baseline).
        truncate_frac : |v_full-v_empty| 의 이 비율보다 개선이 작아지면 순열 조기절단.
        antithetic    : 각 순열을 (π, reversed(π)) 쌍으로 평가해 분산 감소.
        """
        super().__init__(model, peft_config, adapter_name)
        self.loss_fn = loss_fn
        self.n_perm = n_perm
        self.truncate_frac = truncate_frac
        self.group_by = group_by
        self.antithetic = antithetic
        self._rng = np.random.default_rng(seed)
        self.verbose = verbose

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    # ---- lora_E 파라미터 수집 ----
    def _lora_E_params(self, model):
        out = {}
        for n, p in model.named_parameters():
            if f"lora_E.{self.adapter_name}" in n:
                out[n] = p
        return out

    # ---- 살아있는 삼중항 + 모듈별 인덱스 스냅샷 ----
    def _collect(self, Eparams, tol=1e-8):
        orig, mod_idxs = {}, {}
        for name, p in Eparams.items():
            col = p.detach().reshape(-1)
            alive = [i for i in range(col.shape[0]) if abs(float(col[i])) > tol]
            if alive:
                mod_idxs[name] = alive
                for i in alive:
                    orig[(name, i)] = float(col[i])
        return orig, mod_idxs

    # ---- truncated + antithetic MC permutation (공통) ----
    def _run_permutations(self, players, value, v_full, v_empty, tol):
        phi = {p: 0.0 for p in players}
        if self.antithetic:
            n_pairs = max(1, self.n_perm // 2)
            total = 2 * n_pairs
            for t in range(n_pairs):
                base = list(players); self._rng.shuffle(base)
                for perm in (base, list(reversed(base))):
                    active, v_prev = set(), v_empty
                    for p in perm:
                        if abs(v_full - v_prev) < tol:
                            marg = 0.0
                        else:
                            active.add(p)
                            v_cur = value(frozenset(active))
                            marg = v_cur - v_prev; v_prev = v_cur
                        phi[p] += marg
                if self.verbose:
                    print(f"[shapley-alloc] pair {t+1}/{n_pairs}", flush=True)
        else:
            total = self.n_perm
            for t in range(self.n_perm):
                perm = list(players); self._rng.shuffle(perm)
                active, v_prev = set(), v_empty
                for p in perm:
                    if abs(v_full - v_prev) < tol:
                        marg = 0.0
                    else:
                        active.add(p)
                        v_cur = value(frozenset(active))
                        marg = v_cur - v_prev; v_prev = v_cur
                    phi[p] += marg
                if self.verbose:
                    print(f"[shapley-alloc] perm {t+1}/{self.n_perm}", flush=True)
        return {p: phi[p] / total for p in players}

    # ---- stage-1: 모듈 단위 Shapley phi 계산 ----
    def _module_shapley(self, Eparams, orig, mod_idxs):
        players = list(mod_idxs.keys())

        @torch.no_grad()
        def set_state(active_mods):
            active_mods = set(active_mods)
            for name, idxs in mod_idxs.items():
                on = name in active_mods
                for i in idxs:
                    Eparams[name][i, 0] = orig[(name, i)] if on else 0.0

        @torch.no_grad()
        def value(active_mods):
            set_state(frozenset(active_mods))
            return -float(self.loss_fn())

        v_full  = value(players)
        v_empty = value([])
        tol = abs(v_full - v_empty) * self.truncate_frac
        if self.verbose:
            print(f"[shapley-alloc] stage1 module-shapley players={len(players)} "
                  f"v_full={v_full:.5f} v_empty={v_empty:.5f} tol={tol:.5f} "
                  f"n_perm={self.n_perm} antithetic={self.antithetic}", flush=True)

        phi_mod = self._run_permutations(players, value, v_full, v_empty, tol)

        # 원상복구
        with torch.no_grad():
            for (name, i), v in orig.items():
                Eparams[name][i, 0] = v
        return phi_mod

    # ================================================================
    # mask_to_budget : group_by 에 따라 분기
    # ================================================================
    def mask_to_budget(self, model, budget):
        assert self.loss_fn is not None, \
            "ShapleyRankAllocator.loss_fn 이 안 꽂혔다. run()에서 set_loss_fn 호출 필요."

        if self.group_by == "triplet":
            return self._mask_triplet(model, budget)
        if self.group_by == "module":
            return self._mask_module_binary(model, budget)
        return self._mask_hybrid(model, budget)   # 기본

    # ---- HYBRID: stage-1 모듈 Shapley + stage-2 예산 비례배분 + 내부 |λ| top-k ----
    def _mask_hybrid(self, model, budget):
        Eparams = self._lora_E_params(model)
        orig, mod_idxs = self._collect(Eparams)
        if not mod_idxs:
            return {}

        # stage-1: 모듈 Shapley
        phi_mod = self._module_shapley(Eparams, orig, mod_idxs)

        # stage-2: budget 을 phi(양수) 에 비례 배분 → 모듈별 정수 rank
        modules = list(mod_idxs.keys())
        cap = {m: len(mod_idxs[m]) for m in modules}        # 모듈이 가질 수 있는 최대 rank
        w = {m: max(0.0, phi_mod[m]) for m in modules}       # 음수 기여 모듈은 0
        wsum = sum(w.values())

        alloc = {m: 0 for m in modules}
        if wsum <= 0:
            # 전부 비양수면 phi 순위로 budget 개 삼중항을 상위 모듈부터 채움
            order = sorted(modules, key=lambda m: phi_mod[m], reverse=True)
            left = budget
            for m in order:
                take = min(cap[m], left)
                alloc[m] = take; left -= take
                if left <= 0:
                    break
        else:
            # 비례 배분 (실수) → floor, cap 클램프
            raw = {m: budget * w[m] / wsum for m in modules}
            for m in modules:
                alloc[m] = min(cap[m], int(np.floor(raw[m])))
            # 남은 예산을 소수부(잔여 우선) 큰 순서로 +1 씩 분배 (cap 여유 있는 모듈만)
            left = budget - sum(alloc.values())
            frac_order = sorted(
                modules,
                key=lambda m: (raw[m] - np.floor(raw[m])),
                reverse=True,
            )
            idx = 0
            while left > 0 and idx < 10 * len(modules):
                m = frac_order[idx % len(modules)]
                if alloc[m] < cap[m]:
                    alloc[m] += 1; left -= 1
                idx += 1

        if self.verbose:
            nz = sum(1 for m in modules if alloc[m] > 0)
            print(f"[shapley-alloc] stage2 budget={budget} "
                  f"modules_alive={nz}/{len(modules)} "
                  f"alloc_sum={sum(alloc.values())}", flush=True)

        # 모듈 내부: |λ| 상위 alloc[m] 개만 남기고 나머지 0
        rank_pattern = {}
        with torch.no_grad():
            for name, p in Eparams.items():
                col = p.data.view(-1)
                keep = alloc.get(name, 0)
                idxs = mod_idxs.get(name, [])
                if keep <= 0 or not idxs:
                    p.data.zero_()
                    rank_pattern[name] = [False] * col.shape[0]
                    continue
                # 살아있는 삼중항을 |λ| 내림차순 정렬 → 상위 keep 개 유지
                ranked = sorted(idxs, key=lambda i: abs(orig[(name, i)]), reverse=True)
                keep_set = set(ranked[:keep])
                mask = torch.zeros(col.shape[0], dtype=torch.bool, device=col.device)
                for i in range(col.shape[0]):
                    if i in keep_set:
                        mask[i] = True
                    else:
                        col[i] = 0.0
                rank_pattern[name] = mask.tolist()
        return rank_pattern

    # ---- MODULE (binary): 모듈 통째 on/off (rank 8 or 0) ----
    def _mask_module_binary(self, model, budget):
        Eparams = self._lora_E_params(model)
        orig, mod_idxs = self._collect(Eparams)
        if not mod_idxs:
            return {}
        phi_mod = self._module_shapley(Eparams, orig, mod_idxs)

        # phi 상위 모듈부터 통째로 켜서 budget(삼중항 수) 채움
        modules = sorted(mod_idxs.keys(), key=lambda m: phi_mod[m], reverse=True)
        left, on = budget, set()
        for m in modules:
            c = len(mod_idxs[m])
            if left >= c:
                on.add(m); left -= c
        rank_pattern = {}
        with torch.no_grad():
            for name, p in Eparams.items():
                col = p.data.view(-1)
                if name in on:
                    rank_pattern[name] = [i in mod_idxs[name] for i in range(col.shape[0])]
                else:
                    col.zero_()
                    rank_pattern[name] = [False] * col.shape[0]
        return rank_pattern

    # ---- TRIPLET: 전삼중항 개별 Shapley (baseline, 느림) ----
    def _mask_triplet(self, model, budget):
        Eparams = self._lora_E_params(model)
        orig, mod_idxs = self._collect(Eparams)
        players = [(name, i) for name, idxs in mod_idxs.items() for i in idxs]
        if not players:
            return {}

        @torch.no_grad()
        def set_state(active):
            active = set(active)
            for (name, i) in players:
                Eparams[name][i, 0] = orig[(name, i)] if (name, i) in active else 0.0

        @torch.no_grad()
        def value(active):
            set_state(frozenset(active))
            return -float(self.loss_fn())

        v_full  = value(players)
        v_empty = value([])
        tol = abs(v_full - v_empty) * self.truncate_frac
        if self.verbose:
            print(f"[shapley-alloc] triplet players={len(players)} "
                  f"v_full={v_full:.5f} v_empty={v_empty:.5f} tol={tol:.5f}",
                  flush=True)
        phi = self._run_permutations(players, value, v_full, v_empty, tol)

        with torch.no_grad():
            for (name, i), v in orig.items():
                Eparams[name][i, 0] = v

        # 전역 threshold (원본 peft 방식과 동일)
        scores = {}
        for name, p in Eparams.items():
            s = torch.zeros_like(p.data).view(-1)
            for i in range(s.shape[0]):
                s[i] = phi.get((name, i), -1e9)
            scores[name] = s.view(-1, 1)

        all_score = torch.cat([s.view(-1) for s in scores.values()])
        mask_threshold = torch.kthvalue(all_score, k=self.init_bgt - budget)[0].item()
        rank_pattern = {}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if f"lora_E.{self.adapter_name}" in n:
                    p.masked_fill_(scores[n] <= mask_threshold, 0.0)
                    rank_pattern[n] = (~(scores[n] <= mask_threshold)).view(-1).tolist()
        return rank_pattern
"""
추론 latency / peak memory / trainable params 측정 — 재학습 불필요, test 시 측정.

사용법 (run_adalora.py의 test() 안):
  1) load_model 로 epoch4(또는 LoRA) 체크포인트를 이미 로드한 상태에서,
  2) 기존
        with torch.no_grad():
            for history, future, video_user_info in dataloader_test:
                ...
                notebook.record(...)
            notebook.write(result_path)
     블록을 아래 run_test_with_benchmark(...) 호출로 교체하거나,
     이 파일을 viewport_prediction/ 에 두고 test()에서 import 해서 쓰면 됩니다.

핵심 주의:
  - CUDA는 비동기라 반드시 torch.cuda.synchronize()로 커널 완료를 기다린 뒤 시간을 재야 함.
  - 첫 몇 회는 CUDA 초기화/캐싱 때문에 느리므로 warmup 후 측정.
  - LoRA와 AdaLoRA를 같은 GPU에서, 다른 무거운 작업 없이 연달아 재야 공정.
"""
import os
import time
import numpy as np
import pandas as pd
import torch

from utils.normalize import normalize_data, denormalize_data


def count_params(model):
    trainable = total = 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return trainable, total


@torch.no_grad()
def run_test_with_benchmark(args, pipeline, dataloader_test, notebook,
                            result_path, results_dir, file_prefix,
                            warmup=10, tag=None):
    """
    기존 test 루프 + latency/memory/params 측정을 한 번의 패스로 수행.
    notebook.record/write 로 MAE/RMSE도 그대로 기록하고,
    벤치마크 결과는 *_bench.csv 로 저장한다.
    """
    device = args.device
    use_cuda = ('cuda' in str(device)) and torch.cuda.is_available()
    pipeline.eval()

    # ---------- warmup (측정에서 제외) ----------
    it = iter(dataloader_test)
    for _ in range(warmup):
        try:
            history, future, video_user_info = next(it)
        except StopIteration:
            it = iter(dataloader_test)
            history, future, video_user_info = next(it)
        history, future = history.to(device), future.to(device)
        history = normalize_data(history, args.train_dataset)
        pipeline.inference(history, future, video_user_info)
    if use_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)   # warmup 이후로 peak 리셋

    # ---------- 측정 루프 (MAE/RMSE도 함께 기록) ----------
    latencies = []
    for history, future, video_user_info in dataloader_test:
        history, future = history.to(device), future.to(device)
        history = normalize_data(history, args.train_dataset)

        if use_cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        pred, gt = pipeline.inference(history, future, video_user_info)
        if use_cuda:
            torch.cuda.synchronize(device)          # 커널 완료까지 대기 후 측정
        latencies.append((time.perf_counter() - t0) * 1000.0)   # ms

        # --- 기존 정확도 기록 로직 그대로 ---
        pred = denormalize_data(pred, args.test_dataset)
        videos = torch.IntTensor([int(video_user_info[0])])
        users = torch.IntTensor([int(video_user_info[1])])
        timesteps = torch.IntTensor([int(video_user_info[2])])
        notebook.record(pred, gt, videos, users, timesteps)

    notebook.write(result_path)
    print("show detail result:")
    notebook.write_detail(result_path)

    # ---------- 집계 ----------
    lat = np.asarray(latencies)
    trainable, total = count_params(pipeline)
    res = {
        'tag': tag if tag is not None else f'rank_{args.rank}',
        'n_samples': int(lat.size),
        'lat_mean_ms': round(float(lat.mean()), 4),
        'lat_median_ms': round(float(np.median(lat)), 4),
        'lat_p95_ms': round(float(np.percentile(lat, 95)), 4),
        'lat_p99_ms': round(float(np.percentile(lat, 99)), 4),
        'lat_std_ms': round(float(lat.std()), 4),
        'throughput_sps': round(float(1000.0 / lat.mean()), 3),
        'trainable_params': int(trainable),
        'total_params': int(total),
        'trainable_pct': round(100.0 * trainable / total, 4),
    }
    if use_cuda:
        res['peak_mem_MB'] = round(torch.cuda.max_memory_allocated(device) / 1024**2, 3)

    print('\n===== inference benchmark =====')
    for k, v in res.items():
        print(f'  {k:18s}: {v}')

    # 여러 모델을 이어붙일 수 있게 append 모드로 저장
    bench_path = os.path.join(results_dir, file_prefix + '_bench.csv')
    df = pd.DataFrame([res])
    if os.path.exists(bench_path):
        df.to_csv(bench_path, mode='a', header=False, index=False)
    else:
        df.to_csv(bench_path, index=False)
    print(f'saved: {bench_path}')
    return res
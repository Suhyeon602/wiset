import re
import matplotlib.pyplot as plt


def parse_logs_from_file(file_path):
    """로그 파일에서 global_step과 average loss를 추출하는 함수"""
    steps = []
    losses = []
    pattern = re.compile(r'global_step\s+(\d+),\s+average loss:\s+([0-9.]+)')

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    steps.append(int(match.group(1)))
                    losses.append(float(match.group(2)))
    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
        return [], []

    return steps, losses


# 실제 로그 파일명
lora_file = "data/ft_plms/gpt2_base_low_rank/freeze_plm_False/Jin2022/5Hz/his_10_fut_20_ss_15_epochs_5_bs_32_lr_0.0002_seed_1_rank_32_scheduled_sampling_True_console.log"
adalora_file = "data/ft_plms/gpt2_base_adalora/freeze_plm_False/Jin2022/5Hz/his_10_fut_20_ss_15_epochs_5_bs_32_lr_0.0002_seed_1_rank_32_scheduled_sampling_True_console.log"

print("로그 파일을 분석 중입니다...")
steps_lora, losses_lora = parse_logs_from_file(lora_file)
steps_adalora, losses_adalora = parse_logs_from_file(adalora_file)

if steps_lora and steps_adalora:
    # 전체 스텝 중 가장 큰 값을 찾습니다.
    max_step = max(max(steps_lora), max(steps_adalora))
    chunk_size = 10000

    # 0부터 max_step까지 10000 단위로 반복합니다.
    for start_step in range(0, max_step, chunk_size):
        end_step = start_step + chunk_size

        # 현재 구간에 해당하는 데이터만 필터링
        chunk_steps_lora = [s for s in steps_lora if start_step < s <= end_step]
        chunk_losses_lora = [l for s, l in zip(steps_lora, losses_lora) if start_step < s <= end_step]

        chunk_steps_adalora = [s for s in steps_adalora if start_step < s <= end_step]
        chunk_losses_adalora = [l for s, l in zip(steps_adalora, losses_adalora) if start_step < s <= end_step]

        # 해당 구간에 데이터가 존재할 때만 그래프 생성
        if chunk_steps_lora or chunk_steps_adalora:
            plt.figure(figsize=(14, 7))

            if chunk_steps_lora:
                plt.plot(chunk_steps_lora, chunk_losses_lora, label='LoRA Loss', color='blue', alpha=0.8, linewidth=1.5)
            if chunk_steps_adalora:
                plt.plot(chunk_steps_adalora, chunk_losses_adalora, label='AdaLoRA Loss', color='red', alpha=0.8,
                         linewidth=1.5)

            # 그래프 서식
            plt.title(f'Training Average Loss ({start_step} ~ {end_step} Steps)', fontsize=16)
            plt.xlabel('Global Step', fontsize=12)
            plt.ylabel('Average Loss', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)

            # 주의: 구간별로 Loss 값 편차가 커서 자동 스케일링이 보기 좋을 수 있습니다.
            # 범위를 고정하고 싶다면 아래 주석을 해제하세요.
            # if start_step == 0:
            #     plt.ylim(0, 0.3)

            plt.tight_layout()

            # 구간을 파일명에 포함하여 저장
            filename = f'loss_comparison_{start_step}_to_{end_step}.png'
            plt.savefig(filename, dpi=300)

            # 메모리 누수를 방지하기 위해 사용한 figure를 닫아줍니다.
            plt.close()

            print(f"그래프가 '{filename}' 파일로 저장되었습니다!")
else:
    print("그래프를 그릴 데이터가 충분하지 않습니다.")
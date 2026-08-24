import re
import matplotlib.pyplot as plt
import os


def parse_logs_from_file(file_path):
    """로그 파일에서 global_step과 average loss를 직접 추출하는 함수"""
    steps = []
    losses = []
    # 'global_step [숫자], average loss: [숫자]' 패턴
    pattern = re.compile(r'global_step\s+(\d+),\s+average loss:\s+([0-9.]+)')

    try:
        # 파일을 읽기 모드(r)로 엽니다.
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    steps.append(int(match.group(1)))
                    losses.append(float(match.group(2)))
    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return [], []

    return steps, losses


# 읽어올 파일의 정확한 이름 (스크립트와 같은 폴더에 있다고 가정)
lora_file = "data/ft_plms/gpt2_base_low_rank/freeze_plm_False/Jin2022/5Hz/his_10_fut_20_ss_15_epochs_5_bs_32_lr_0.0002_seed_1_rank_32_scheduled_sampling_True_console.log"
adalora_file = "data/ft_plms/gpt2_base_adalora/freeze_plm_False/Jin2022/5Hz/his_10_fut_20_ss_15_epochs_5_bs_32_lr_0.0002_seed_1_rank_32_scheduled_sampling_True_console.log"

# 파일에서 데이터 추출
print("로그 파일을 분석 중입니다...")
steps_lora, losses_lora = parse_logs_from_file(lora_file)
steps_adalora, losses_adalora = parse_logs_from_file(adalora_file)

# 두 파일 모두 정상적으로 읽혔을 때만 그래프 출력
if steps_lora and steps_adalora:
    plt.figure(figsize=(14, 7))

    # 꺾은선 그래프 생성
    plt.plot(steps_lora, losses_lora, label='LoRA Loss', color='blue', alpha=0.8, linewidth=1.5)
    plt.plot(steps_adalora, losses_adalora, label='AdaLoRA Loss', color='red', alpha=0.8, linewidth=1.5)

    # 그래프 서식 지정
    plt.title('Training Average Loss Comparison: LoRA vs AdaLoRA', fontsize=16)
    plt.xlabel('Global Step', fontsize=12)
    plt.ylabel('Average Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 초기 Loss 값이 커서 그래프가 눌려 보일 경우 y축 범위 제한
    plt.ylim(0, 0.3)

    plt.tight_layout()

    # ❌ 변경 전: plt.show()
    # ✅ 변경 후: 파일로 저장하도록 수정
    plt.savefig('loss_comparison.png', dpi=300)
    print("그래프가 'loss_comparison.png' 파일로 저장되었습니다!")

else:
    print("그래프를 그릴 데이터가 충분하지 않습니다.")
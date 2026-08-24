import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 파일 이름 정의
file_lora = "data/results/gpt2_base_low_rank/freeze_plm_False/Jin2022/5Hz/his_10_fut_20_ss_15_epochs_5_bs_32_lr_0.0002_seed_1_rank_32_scheduled_sampling_True_results.csv"
file_adalora = "data/results/gpt2_base_adalora/freeze_plm_False/Jin2022/5Hz/his_10_fut_20_ss_15_epochs_5_bs_32_lr_0.0002_seed_1_rank_32_scheduled_sampling_True_results.csv"

print("데이터를 불러오는 중입니다...")
# 2. CSV 읽기 및 불필요한 'Unnamed' 컬럼 방지
df_lora = pd.read_csv(file_lora)
df_lora = df_lora.loc[:, ~df_lora.columns.str.contains('^Unnamed')]

df_adalora = pd.read_csv(file_adalora)
df_adalora = df_adalora.loc[:, ~df_adalora.columns.str.contains('^Unnamed')]

# 3. 모델명 태그 추가
df_lora['model'] = 'LoRA'
df_adalora['model'] = 'AdaLoRA'

# 4. 두 데이터프레임 병합 (형태가 똑같으므로 바로 합치면 됩니다)
df_combined = pd.concat([df_lora, df_adalora], ignore_index=True)

# 5. 전체 통계량 요약 출력
print("\n====== 전체 성능 비교 (Overall Performance) ======")
summary = df_combined.groupby('model')[['mae', 'rmse']].mean().round(3)
print(summary)
print("\n")

print("====== 비디오별 평균 MAE 비교 ======")
video_summary = df_combined.pivot_table(index='video', columns='model', values='mae', aggfunc='mean').round(3)
video_summary['Diff (Ada-LoRA)'] = video_summary['AdaLoRA'] - video_summary['LoRA']
print(video_summary)

# 6. 시각화 1: 모델별 전체 오차 분포 (Boxplot)
print("\n그래프를 생성하고 있습니다...")
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x='model', y='mae', data=df_combined, palette='Set2')
plt.title('MAE Distribution: LoRA vs AdaLoRA')

plt.subplot(1, 2, 2)
sns.boxplot(x='model', y='rmse', data=df_combined, palette='Set2')
plt.title('RMSE Distribution: LoRA vs AdaLoRA')
plt.tight_layout()
plt.savefig('error_distribution_boxplot.png', dpi=300)
plt.close()

# 7. 시각화 2: 비디오별 MAE 비교 (Barplot)
plt.figure(figsize=(12, 6))
sns.barplot(x='video', y='mae', hue='model', data=df_combined, palette='Set1', errorbar=None)
plt.title('Average MAE by Video ID', fontsize=14)
plt.xlabel('Video ID')
plt.ylabel('Average MAE')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Model')
plt.tight_layout()
plt.savefig('mae_by_video_barplot.png', dpi=300)
plt.close()

print("\n비교 분석이 완료되었습니다! 디렉토리에서 'error_distribution_boxplot.png'와 'mae_by_video_barplot.png'를 확인해주세요.")
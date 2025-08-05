import pandas as pd
from src.rfm_a_model import RFMAModel
from src.visualization import (
    plot_segment_distribution,
    plot_rfm_radar,
    plot_adherence_analysis,
    plot_segment_comparison
)
from src.deployment import RFMADeployment
import os

# 检查数据是否存在，如果不存在则生成
if not os.path.exists("data/patients.csv"):
    print("Generating test data...")
    from src.data_generator import generate_all_data
    generate_all_data()

# 加载数据
print("Loading data...")
patients_df = pd.read_csv("data/patients.csv")
transactions_df = pd.read_csv("data/transactions.csv")
prescriptions_df = pd.read_csv("data/prescriptions.csv")
adherence_df = pd.read_csv("data/adherence.csv")

# 初始化并运行RFM-A模型
print("Running RFM-A model...")
rfma_model = RFMAModel(patients_df, transactions_df, prescriptions_df, adherence_df)
rfma_model.segment_patients()

# 获取结果
results_df = rfma_model.results_df
segment_counts = rfma_model.get_segment_counts()

# 保存结果
results_df.to_csv("rfma_results.csv", index=False)
print("Results saved to rfma_results.csv")

# 可视化分析
print("Generating visualizations...")
plot_segment_distribution(segment_counts)
plot_rfm_radar(results_df)
plot_adherence_analysis(results_df)
plot_segment_comparison(results_df)

# 部署示例
print("Generating action plans...")
deployment = RFMADeployment(rfma_model)

# 为前5位高风险患者生成行动建议
high_risk_patients = rfma_model.get_high_risk_patients().head(5)
for _, patient in high_risk_patients.iterrows():
    action_plan = deployment.generate_actions(patient['patient_id'])
    print("\nAction Plan for High-Risk Patient:")
    print(f"Patient: {patient['name']} (ID: {patient['patient_id']})")
    print(f"Disease: {patient['primary_disease']}")
    print(f"Adherence Score: {patient['adherence_score']:.2f}")
    print("Recommended Actions:")
    for i, action in enumerate(action_plan['actions'], 1):
        print(f"  {i}. {action}")

# 生成所有患者的行动建议
all_actions_df = deployment.generate_all_actions()
all_actions_df.to_csv("patient_action_plans.csv", index=False)
print("\nAll patient action plans saved to patient_action_plans.csv")

print("\nRFM-A analysis completed successfully!")

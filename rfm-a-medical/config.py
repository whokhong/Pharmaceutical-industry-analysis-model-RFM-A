# 模型参数配置
MODEL_CONFIG = {
    "weights": {
        "recency": 0.2,
        "frequency": 0.2,
        "monetary": 0.2,
        "adherence": 0.4
    },
    "score_bins": 5,
    "segmentation": {
        "high_value": {"min_score": 4.5, "min_adherence": 4},
        "high_risk": {"min_score": 3.0, "max_adherence": 2},
        "churn_risk": {"max_recency": 1, "min_adherence": 4},
        "low_value": {"max_score": 2.0}
    }
}

# 数据生成参数
DATA_GENERATION = {
    "num_patients": 1000,
    "start_date": "2023-01-01",
    "end_date": "2024-06-30",
    "disease_types": [
        "Diabetes", "Hypertension", "Asthma", 
        "Hyperlipidemia", "Arthritis", "Thyroid"
    ],
    "drug_categories": {
        "Diabetes": ["Insulin", "Metformin", "GLP-1"],
        "Hypertension": ["ACE Inhibitors", "Beta Blockers"],
        "Asthma": ["Inhalers", "Corticosteroids"],
        "Hyperlipidemia": ["Statins"],
        "Arthritis": ["NSAIDs", "DMARDs"],
        "Thyroid": ["Levothyroxine"]
    },
    "adherence_factors": {
        "age_effect": 0.5,  # 年龄对依从性的影响系数
        "disease_effect": {
            "Diabetes": -0.3, 
            "Hypertension": -0.2,
            "Asthma": -0.1,
            "Hyperlipidemia": -0.15,
            "Arthritis": -0.1,
            "Thyroid": -0.05
        }
    }
}

import pandas as pd
import numpy as np
from datetime import datetime
from rfm_a_model import RFMAModel

class RFMADeployment:
    def __init__(self, model):
        self.model = model
        self.actions = {
            "高价值患者": self.high_value_action,
            "高风险患者": self.high_risk_action,
            "流失预警": self.churn_risk_action,
            "低价值群体": self.low_value_action,
            "普通患者": self.general_action
        }
    
    def generate_actions(self, patient_id):
        """为指定患者生成行动建议"""
        if self.model.results_df is None:
            self.model.segment_patients()
            
        patient_data = self.model.results_df[
            self.model.results_df['patient_id'] == patient_id
        ]
        
        if patient_data.empty:
            return {"error": "Patient not found"}
        
        segment = patient_data['segment'].values[0]
        return self.actions[segment](patient_data)
    
    def high_value_action(self, patient_data):
        """高价值患者行动建议"""
        patient = patient_data.iloc[0]
        return {
            "patient_id": patient['patient_id'],
            "name": patient['name'],
            "segment": "高价值患者",
            "actions": [
                "邀请加入VIP健康管理计划",
                "提供专属健康顾问服务",
                "发放年度健康体检优惠券",
                "推送个性化健康资讯"
            ],
            "priority": "高",
            "follow_up_days": 30
        }
    
    def high_risk_action(self, patient_data):
        """高风险患者行动建议"""
        patient = patient_data.iloc[0]
        disease = patient['primary_disease']
        
        actions = [
            "药师3天内电话随访",
            "发送用药指导手册",
            "安排免费健康检测"
        ]
        
        if disease == "Diabetes":
            actions.append("推送血糖管理指南")
        elif disease == "Hypertension":
            actions.append("提供血压监测设备优惠")
        
        return {
            "patient_id": patient['patient_id'],
            "name": patient['name'],
            "segment": "高风险患者",
            "actions": actions,
            "priority": "紧急",
            "follow_up_days": 7
        }
    
    def churn_risk_action(self, patient_data):
        """流失预警患者行动建议"""
        patient = patient_data.iloc[0]
        last_purchase_days = patient['recency_days']
        
        discount = min(30, max(5, int(last_purchase_days / 10)))  # 5%-30%折扣
        
        return {
            "patient_id": patient['patient_id'],
            "name": patient['name'],
            "segment": "流失预警",
            "actions": [
                f"发送{discount}%折扣优惠券",
                "推送个性化用药提醒",
                "提供免费送药上门服务"
            ],
            "priority": "中",
            "follow_up_days": 14
        }
    
    def low_value_action(self, patient_data):
        """低价值群体行动建议"""
        patient = patient_data.iloc[0]
        return {
            "patient_id": patient['patient_id'],
            "name": patient['name'],
            "segment": "低价值群体",
            "actions": [
                "推送健康知识科普",
                "发送新用户优惠包",
                "邀请参加健康讲座"
            ],
            "priority": "低",
            "follow_up_days": 60
        }
    
    def general_action(self, patient_data):
        """普通患者行动建议"""
        patient = patient_data.iloc[0]
        return {
            "patient_id": patient['patient_id'],
            "name": patient['name'],
            "segment": "普通患者",
            "actions": [
                "推送常规健康提示",
                "发送会员积分提醒"
            ],
            "priority": "常规",
            "follow_up_days": 90
        }
    
    def generate_all_actions(self):
        """为所有患者生成行动建议"""
        if self.model.results_df is None:
            self.model.segment_patients()
            
        actions_list = []
        for _, row in self.model.results_df.iterrows():
            action = self.actions[row['segment']](pd.DataFrame([row]))
            actions_list.append(action)
        
        return pd.DataFrame(actions_list)

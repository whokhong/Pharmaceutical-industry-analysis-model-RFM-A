import pandas as pd
import numpy as np
from datetime import datetime
from config import MODEL_CONFIG

class RFMAModel:
    def __init__(self, patients_df, transactions_df, prescriptions_df, adherence_df):
        """
        初始化RFM-A模型
        
        参数:
            patients_df: 患者数据DataFrame
            transactions_df: 交易数据DataFrame
            prescriptions_df: 处方数据DataFrame
            adherence_df: 依从性数据DataFrame
        """
        self.patients_df = patients_df
        self.transactions_df = transactions_df
        self.prescriptions_df = prescriptions_df
        self.adherence_df = adherence_df
        self.results_df = None
        self.segmentation_counts = None
        
    def preprocess_data(self, end_date=None):
        """数据预处理"""
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')
        
        # 计算Recency
        recency_df = self.transactions_df.groupby('patient_id')['transaction_date'].max().reset_index()
        recency_df['recency_days'] = (pd.to_datetime(end_date) - pd.to_datetime(recency_df['transaction_date'])).dt.days
        
        # 计算Frequency
        frequency_df = self.transactions_df.groupby('patient_id')['transaction_id'].count().reset_index()
        frequency_df.rename(columns={'transaction_id': 'order_count'}, inplace=True)
        
        # 计算Monetary
        monetary_df = self.transactions_df.groupby('patient_id')['amount'].sum().reset_index()
        monetary_df.rename(columns={'amount': 'total_spent'}, inplace=True)
        
        # 合并RFM数据
        rfm_df = recency_df.merge(frequency_df, on='patient_id')
        rfm_df = rfm_df.merge(monetary_df, on='patient_id')
        
        # 添加Adherence数据
        self.rfma_df = rfm_df.merge(self.adherence_df, on='patient_id')
        
        return self.rfma_df
    
    def calculate_scores(self):
        """计算RFM-A分数"""
        if self.rfma_df is None:
            self.preprocess_data()
            
        df = self.rfma_df.copy()
        n_bins = MODEL_CONFIG["score_bins"]
        
        # Recency评分 (天数越少分数越高)
        df['R_score'] = pd.qcut(
            df['recency_days'], 
            q=n_bins, 
            labels=range(n_bins, 0, -1),
            duplicates='drop'
        ).astype(int)
        
        # Frequency评分
        df['F_score'] = pd.qcut(
            df['order_count'], 
            q=n_bins, 
            labels=range(1, n_bins+1),
            duplicates='drop'
        ).astype(int)
        
        # Monetary评分
        df['M_score'] = pd.qcut(
            df['total_spent'], 
            q=n_bins, 
            labels=range(1, n_bins+1),
            duplicates='drop'
        ).astype(int)
        
        # Adherence评分
        df['A_score'] = pd.qcut(
            df['adherence_score'], 
            q=n_bins, 
            labels=range(1, n_bins+1),
            duplicates='drop'
        ).astype(int)
        
        # 计算加权总分
        weights = MODEL_CONFIG["weights"]
        df['RFMA_score'] = (
            df['R_score'] * weights['recency'] +
            df['F_score'] * weights['frequency'] +
            df['M_score'] * weights['monetary'] +
            df['A_score'] * weights['adherence']
        )
        
        self.scored_df = df
        return df
    
    def segment_patients(self):
        """客户分层"""
        if self.scored_df is None:
            self.calculate_scores()
            
        df = self.scored_df.copy()
        seg_config = MODEL_CONFIG["segmentation"]
        
        # 应用分层规则
        conditions = [
            # 高价值患者
            (df['RFMA_score'] >= seg_config['high_value']['min_score']) & 
            (df['A_score'] >= seg_config['high_value']['min_adherence']),
            
            # 高风险患者
            (df['RFMA_score'] >= seg_config['high_risk']['min_score']) & 
            (df['A_score'] <= seg_config['high_risk']['max_adherence']),
            
            # 流失风险患者
            (df['R_score'] <= seg_config['churn_risk']['max_recency']) & 
            (df['A_score'] >= seg_config['churn_risk']['min_adherence']),
            
            # 低价值患者
            (df['RFMA_score'] <= seg_config['low_value']['max_score'])
        ]
        
        labels = [
            '高价值患者', 
            '高风险患者', 
            '流失预警', 
            '低价值群体'
        ]
        
        df['segment'] = np.select(
            conditions, 
            labels, 
            default='普通患者'
        )
        
        # 添加患者基本信息
        df = df.merge(
            self.patients_df[['patient_id', 'name', 'age', 'gender', 'primary_disease']], 
            on='patient_id'
        )
        
        self.results_df = df
        self.segmentation_counts = df['segment'].value_counts().to_dict()
        
        return df
    
    def get_segment_counts(self):
        """获取各分层的患者数量"""
        if self.segmentation_counts is None:
            self.segment_patients()
        return self.segmentation_counts
    
    def get_high_risk_patients(self):
        """获取高风险患者列表"""
        if self.results_df is None:
            self.segment_patients()
        return self.results_df[self.results_df['segment'] == '高风险患者']
    
    def get_high_value_patients(self):
        """获取高价值患者列表"""
        if self.results_df is None:
            self.segment_patients()
        return self.results_df[self.results_df['segment'] == '高价值患者']
    
    def get_churn_risk_patients(self):
        """获取流失风险患者列表"""
        if self.results_df is None:
            self.segment_patients()
        return self.results_df[self.results_df['segment'] == '流失预警']
    
    def save_results(self, output_path):
        """保存结果到CSV"""
        if self.results_df is None:
            self.segment_patients()
        self.results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

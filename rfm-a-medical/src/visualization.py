import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from config import MODEL_CONFIG

def plot_segment_distribution(segment_counts, title="客户分层分布"):
    """绘制客户分层分布图"""
    segments = list(segment_counts.keys())
    counts = list(segment_counts.values())
    
    plt.figure(figsize=(10, 6))
    colors = ['#4CAF50', '#FFC107', '#F44336', '#9E9E9E', '#2196F3']
    explode = (0.05, 0.05, 0.05, 0.05, 0.05)
    
    plt.pie(
        counts, 
        labels=segments, 
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        explode=explode,
        shadow=True
    )
    plt.title(title, fontsize=14)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def plot_rfm_radar(results_df):
    """绘制RFM-A雷达图"""
    # 计算各分层的平均得分
    segment_avg = results_df.groupby('segment')[
        ['R_score', 'F_score', 'M_score', 'A_score']
    ].mean().reset_index()
    
    # 设置雷达图参数
    categories = ['Recency', 'Frequency', 'Monetary', 'Adherence']
    N = len(categories)
    
    # 为每个分层绘制雷达图
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    for i, row in segment_avg.iterrows():
        values = row[['R_score', 'F_score', 'M_score', 'A_score']].tolist()
        values += values[:1]  # 闭合雷达图
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['segment'])
        ax.fill(angles, values, alpha=0.1)
    
    # 设置角度轴
    plt.xticks(angles[:-1], categories)
    
    # 设置径向轴
    ax.set_rlabel_position(0)
    plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], color="grey", size=8)
    plt.ylim(0, 5.5)
    
    # 添加图例和标题
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('各客户分层的RFM-A平均得分', size=16, y=1.1)
    plt.tight_layout()
    plt.show()

def plot_adherence_analysis(results_df):
    """绘制依从性分析图"""
    plt.figure(figsize=(12, 8))
    
    # 依从性分布直方图
    plt.subplot(2, 2, 1)
    sns.histplot(results_df['adherence_score'], bins=20, kde=True)
    plt.title('用药依从性分布')
    plt.xlabel('依从性评分')
    plt.ylabel('患者数量')
    
    # 依从性与疾病类型的关系
    plt.subplot(2, 2, 2)
    sns.boxplot(x='primary_disease', y='adherence_score', data=results_df)
    plt.title('不同疾病的依从性比较')
    plt.xlabel('疾病类型')
    plt.ylabel('依从性评分')
    plt.xticks(rotation=45)
    
    # 依从性与年龄的关系
    plt.subplot(2, 2, 3)
    sns.regplot(x='age', y='adherence_score', data=results_df, 
               scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    plt.title('年龄与依从性的关系')
    plt.xlabel('年龄')
    plt.ylabel('依从性评分')
    
    # 依从性与消费金额的关系
    plt.subplot(2, 2, 4)
    sns.scatterplot(x='adherence_score', y='total_spent', 
                   hue='segment', data=results_df, alpha=0.6)
    plt.title('依从性与消费金额的关系')
    plt.xlabel('依从性评分')
    plt.ylabel('消费金额')
    
    plt.tight_layout()
    plt.show()

def plot_segment_comparison(results_df):
    """绘制各分层特征比较"""
    plt.figure(figsize=(14, 10))
    
    # Recency比较
    plt.subplot(2, 2, 1)
    sns.boxplot(x='segment', y='recency_days', data=results_df)
    plt.title('各分层最近购买间隔(天)')
    plt.xlabel('客户分层')
    plt.ylabel('最近购买天数')
    
    # Frequency比较
    plt.subplot(2, 2, 2)
    sns.boxplot(x='segment', y='order_count', data=results_df)
    plt.title('各分层购买频率')
    plt.xlabel('客户分层')
    plt.ylabel('购买次数')
    
    # Monetary比较
    plt.subplot(2, 2, 3)
    sns.boxplot(x='segment', y='total_spent', data=results_df)
    plt.title('各分层消费金额')
    plt.xlabel('客户分层')
    plt.ylabel('消费金额')
    
    # Adherence比较
    plt.subplot(2, 2, 4)
    sns.boxplot(x='segment', y='adherence_score', data=results_df)
    plt.title('各分层用药依从性')
    plt.xlabel('客户分层')
    plt.ylabel('依从性评分')
    
    plt.tight_layout()
    plt.show()

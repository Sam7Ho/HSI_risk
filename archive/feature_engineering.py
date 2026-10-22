"""
feature_engineering.py - 特徵工程
功能：從原始數據產生特徵
"""

import numpy as np
import pandas as pd


def create_features(df):
    """產生特徵"""
    df = df.copy()
    
    # 價格特徵
    df['HSI_Return'] = df['HSI_Close'].pct_change()
    df['Volatility_20'] = df['HSI_Return'].rolling(window=20).std() * np.sqrt(252)
    df['Volatility_60'] = df['HSI_Return'].rolling(window=60).std() * np.sqrt(252)
    df['SMA_20'] = df['HSI_Close'].rolling(window=20).mean()
    df['SMA_60'] = df['HSI_Close'].rolling(window=60).mean()
    df['Dist_SMA_20'] = (df['HSI_Close'] - df['SMA_20']) / df['SMA_20']
    df['Dist_SMA_60'] = (df['HSI_Close'] - df['SMA_60']) / df['SMA_60']
    df['Trend_Up'] = (df['SMA_20'] > df['SMA_60']).astype(int)
    
    # VHSI 特徵
    df['VHSI_Change'] = df['VHSI_Close'].pct_change()
    df['VHSI_SMA_10'] = df['VHSI_Close'].rolling(window=10).mean()
    df['VHSI_ZScore'] = (
        (df['VHSI_Close'] - df['VHSI_Close'].rolling(60).mean()) 
        / df['VHSI_Close'].rolling(60).std()
    )
    
    # 清理
    df = df.dropna()
    
    print(f"✅ 特徵完成，共 {len(df)} 筆")
    return df


if __name__ == "__main__":
    from data_preparation import load_data
    
    df = load_data()
    df = create_features(df)
    df.to_csv("data/featured_data.csv")
    print("💾 已儲存：data/featured_data.csv")
    # 加在 create_features 最後
    print(f"\n📊 特徵檢查：")
    print(f"   特徵數：{df.shape[1]}")
    print(f"   缺失值：{df.isnull().sum().sum()}")
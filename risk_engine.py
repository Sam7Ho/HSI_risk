"""
risk_engine.py - 尾部風險評估引擎 (EVT / POT 方法)
目標：計算當前市場的尾部風險（VaR / CVaR），而非預測何時崩盤。
"""

import pandas as pd
import numpy as np
from scipy.stats import genpareto
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.ensemble import IsolationForest

# 設定中文字型 (Mac 專用)
try:
    fm.fontManager.addfont("/Library/Fonts/Arial Unicode.ttf")
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

def calculate_evt_risk(returns, threshold_percentile=95):
    """
    使用 POT (Peaks-Over-Threshold) 方法計算尾部風險
    
    Args:
        returns: 每日報酬率序列 (Series)
        threshold_percentile: 閾值百分位數 (預設 95%，即只看最差的 5% 虧損)
        
    Returns:
        dict: 包含 VaR, CVaR 等資訊
    """
    # 只保留虧損 (負報酬)
    losses = -returns[returns < 0]
    
    if len(losses) < 30:
        return None

    # 1. 設定閾值 (Threshold, u)
    threshold = np.percentile(losses, threshold_percentile)
    
    # 2. 提取超額虧損 (Excesses, y = x - u)
    # 我們只關心超過閾值的極端虧損
    excesses = losses[losses > threshold] - threshold
    
    if len(excesses) < 10:
        return None

    # 3. 擬合廣義帕雷托分佈 (GPD)
    # shape (c) 和 scale 是 GPD 的參數
    c, loc, scale = genpareto.fit(excesses, floc=0)
    
    # 4. 計算 VaR 和 CVaR (以 99% 信心水準為例)
    p = 0.99
    n = len(losses)
    n_u = len(excesses)
    
    # 計算 VaR
    if c != 0:
        var_99 = threshold + (scale / c) * (((n / n_u) * (1 - p)) ** (-c) - 1)
        # 計算 CVaR (Expected Shortfall)
        cvar_99 = var_99 / (1 - c) + (scale - c * threshold) / (1 - c)
    else:
        var_99 = threshold - scale * np.log((n / n_u) * (1 - p))
        cvar_99 = var_99 + scale

    return {
        'threshold': threshold,
        'var_99': var_99,
        'cvar_99': cvar_99,
        'gpd_shape': c,
        'gpd_scale': scale
    }

def run_risk_assessment():
    print("="*60)
    print("🛡️ 市場尾部風險評估 (EVT / POT 模型)")
    print("="*60)
    
    # 1. 載入數據
    df = pd.read_csv("data/raw_data.csv", index_col=0, parse_dates=True)
    
    # 2. 計算日報酬率
    df['Return'] = df['HSI_Close'].pct_change()
    df = df.dropna()
    
    # 3. 滾動計算歷史風險 (例如每 60 天更新一次模型)
    # 為了展示，我們計算「當前」最新的風險狀態
    lookback_window = 250 # 用過去一年的數據來建模
    recent_returns = df['Return'].tail(lookback_window)
    
    print(f"\n📅 分析區間：{df.index[-lookback_window].date()} ~ {df.index[-1].date()}")
    print(f"📊 當前 VHSI：{df['VHSI_Close'].iloc[-1]:.2f}")
    print(f"📊 當前價格：{df['HSI_Close'].iloc[-1]:.2f}")
    
    # 4. 計算 EVT 風險
    risk_metrics = calculate_evt_risk(recent_returns, threshold_percentile=90) # 用 90% 增加樣本
    
    if risk_metrics:
        print("\n" + "-"*40)
        print("📉 當前尾部風險評估 (99% 信心水準)")
        print("-"*40)
        print(f"   單日 VaR  : {risk_metrics['var_99']*100:.2f}%")
        print(f"   單日 CVaR : {risk_metrics['cvar_99']*100:.2f}%")
        print(f"   (CVaR 代表：如果真的發生最壞的 1% 情況，平均會跌多少)")
        
        # 5. 風險狀態判定
        current_risk = risk_metrics['var_99']
        if current_risk > 0.04:
            state = "🔴 極端風險 (Extreme)"
        elif current_risk > 0.025:
            state = "🟠 警戒風險 (Elevated)"
        else:
            state = "🟢 正常風險 (Normal)"
            
        print("\n" + "-"*40)
        print(f"🚦 當前風險狀態：{state}")
        print("-"*40)
        
        # 6. 視覺化：畫出尾部分佈
        plot_tail_risk(df, recent_returns, risk_metrics)

def plot_tail_risk(df, returns, risk_metrics):
    """繪製尾部風險圖表"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 圖 1：HSI 價格與 VHSI
    ax1 = axes[0]
    ax1.plot(df.index[-250:], df['HSI_Close'].tail(250), color='#2c3e50', label='HSI 指數')
    ax1.set_ylabel('HSI 指數', color='#2c3e50')
    ax1.tick_params(axis='y', labelcolor='#2c3e50')
    
    ax2 = ax1.twinx()
    ax2.plot(df.index[-250:], df['VHSI_Close'].tail(250), color='#e74c3c', label='VHSI 恐慌指數')
    ax2.set_ylabel('VHSI', color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    
    ax1.set_title('近一年 HSI 走勢與 VHSI 恐慌指數', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 圖 2：虧損分佈直方圖與 VaR 標示
    ax = axes[1]
    losses = -returns[returns < 0] * 100 # 轉為百分比
    
    ax.hist(losses, bins=50, color='#bdc3c7', edgecolor='white', alpha=0.7)
    
    # 畫出 VaR 和 CVaR 線
    var_line = risk_metrics['var_99'] * 100
    cvar_line = risk_metrics['cvar_99'] * 100
    
    ax.axvline(var_line, color='#e67e22', linestyle='--', linewidth=2, label=f'99% VaR ({var_line:.2f}%)')
    ax.axvline(cvar_line, color='#e74c3c', linestyle='-', linewidth=2, label=f'99% CVaR ({cvar_line:.2f}%)')
    
    ax.set_title('過去一年虧損分佈與尾部風險估計', fontsize=14, fontweight='bold')
    ax.set_xlabel('單日虧損幅度 (%)')
    ax.set_ylabel('天數')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/tail_risk_assessment.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n✅ 風險評估圖表已儲存：outputs/tail_risk_assessment.png")


def plot_rolling_var(df, window=250):
    """
    計算並繪製動態滾動 VaR
    Args:
        df: 包含 Return 的 DataFrame
        window: 滾動窗口大小 (預設 250 天)
    """
    print(f"\n🔄 正在計算 {window} 天滾動 VaR (這需要一點時間)...")
    
    rolling_vars = []
    dates = []
    
    # 我們從第 window 天開始算，這樣才有足夠的歷史數據
    for i in range(window, len(df)):
        # 取出過去 window 天的報酬
        past_returns = df['Return'].iloc[i-window:i]
        
        # 計算 EVT 風險
        risk = calculate_evt_risk(past_returns, threshold_percentile=90)
        
        if risk:
            rolling_vars.append(risk['var_99'] * 100) # 轉為 %
            dates.append(df.index[i])
    
    # 畫圖
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 畫 HSI 走勢
    ax.plot(df.index[-len(dates):], df['HSI_Close'].tail(len(dates)), 
            color='#2c3e50', alpha=0.5, label='HSI 指數')
    ax.set_ylabel('HSI 指數', color='#2c3e50')
    ax.tick_params(axis='y', labelcolor='#2c3e50')
    
    # 畫 Rolling VaR
    ax2 = ax.twinx()
    ax2.plot(dates, rolling_vars, color='#e74c3c', linewidth=2, label='250天滾動 99% VaR')
    ax2.axhline(y=2.5, color='#f39c12', linestyle='--', alpha=0.5, label='警戒線 (2.5%)')
    ax2.axhline(y=4.0, color='#c0392b', linestyle='--', alpha=0.5, label='極端風險線 (4.0%)')
    
    ax2.set_ylabel('單日 VaR (%)', color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    
    ax.set_title('歷史動態風險追蹤 (Rolling 99% VaR)', fontsize=16, fontweight='bold')
    
    # 合併圖例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/rolling_var_trend.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ 動態風險圖表已儲存：outputs/rolling_var_trend.png")

def detect_anomalies(df):
    """
    使用 Isolation Forest 檢測異常狀態
    特徵：波動率、VHSI、距離均線的距離
    """
    print("\n🕵️ 正在執行異常檢測 (Isolation Forest)...")
    
    # 1. 準備特徵
    # 我們要看的是：波動率是否異常低 (暴風雨前寧靜)，或者 VHSI 異常
    df_anomaly = df.copy()
    df_anomaly['Volatility_20'] = df_anomaly['Return'].rolling(20).std() * np.sqrt(252)
    df_anomaly['VHSI_Change'] = df_anomaly['VHSI_Close'].pct_change()
    df_anomaly = df_anomaly.dropna()
    
    features = ['Volatility_20', 'VHSI_Close', 'VHSI_Change']
    X = df_anomaly[features]
    
    # 2. 訓練 Isolation Forest
    # contamination=0.02 代表我們預期約有 2% 的極端異常日
    iso_forest = IsolationForest(contamination=0.02, random_state=42)
    df_anomaly['Anomaly'] = iso_forest.fit_predict(X)
    
    # 預測結果：-1 代表異常，1 代表正常
    anomalies = df_anomaly[df_anomaly['Anomaly'] == -1]
    
    print(f"   ✅ 檢測完成，發現 {len(anomalies)} 個異常交易日。")
    
    # 3. 畫出異常點
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_anomaly.index[-500:], df_anomaly['HSI_Close'].tail(500), 
            color='#2c3e50', label='HSI 指數')
    
    # 把異常點圈起來
    anomalies_recent = anomalies[anomalies.index >= df_anomaly.index[-500]]
    ax.scatter(anomalies_recent.index, anomalies_recent['HSI_Close'], 
               color='#e74c3c', marker='x', s=100, label='異常狀態 (Anomaly)', zorder=5)
    
    ax.set_title('歷史異常狀態檢測 (過去 2 年)', fontsize=16, fontweight='bold')
    ax.set_ylabel('HSI 指數')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/anomaly_detection.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ 異常檢測圖表已儲存：outputs/anomaly_detection.png")

if __name__ == "__main__":
    import os
    os.makedirs('outputs', exist_ok=True)
    run_risk_assessment()
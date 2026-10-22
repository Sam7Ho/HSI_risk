"""
main.py - 風險評估主程式
一鍵執行：數據更新 -> 風險計算 -> 輸出報告
"""

from data_preparation import fetch_data, save_data, load_data
from risk_engine import run_risk_assessment, plot_rolling_var, detect_anomalies
import os

def main():
    print("="*60)
    print("🛡️ 港股尾部風險評估系統")
    print("="*60)
    
    # 確保目錄存在
    os.makedirs('data', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # ============================================
    # 步驟 1：更新數據
    # ============================================
    print("\n【步驟 1】數據更新")
    print("-" * 60)
    
    try:
        # 嘗試載入現有數據 (節省時間)
        df = load_data()
    except FileNotFoundError:
        # 如果沒有數據，則重新下載
        df = fetch_data(start_date="2015-01-01")
        save_data(df)
    
    # ============================================
    # 步驟 2：執行風險評估
    # ============================================
    print("\n【步驟 2】尾部風險評估 (EVT)")
    print("-" * 60)
    
    # 這裡直接呼叫 risk_engine.py 裡的函數
    # 它會印出 VaR/CVaR，並產生圖表
    run_risk_assessment()

    # ============================================
    # 步驟 3：動態追蹤
    # ============================================
    print("\n【步驟 3】動態風險追蹤")
    print("-" * 60)
    # 在執行前，先確保 df 有計算 Return
    df['Return'] = df['HSI_Close'].pct_change()
    df = df.dropna()
    plot_rolling_var(df, window=250)

    # ============================================
    # 步驟 4：異常檢測
    # ============================================
    print("\n【步驟 4】異常檢測")
    print("-" * 60)
    detect_anomalies(df)

    print("\n" + "="*60)
    print("✅ 風險評估流程執行完畢")
    print("="*60)
    print("📊 請查看 outputs/")


if __name__ == "__main__":
    main()